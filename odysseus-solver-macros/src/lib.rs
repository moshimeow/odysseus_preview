use proc_macro::TokenStream;
use proc_macro2::{Literal, Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Attribute macro for functions that do arithmetic with a generic `Real` type.
///
/// Within the attributed function, any numeric literal with the suffix `R`
/// (e.g. `2.0R`, `3R`) is rewritten to `T::from_f64(<value>)`, where `T` is
/// the first type parameter that implements `Real`. This avoids the orphan-rule
/// limitation that prevents `f64 * T` from working symmetrically with `T * f64`.
///
/// Example:
///   #[real_fn]
///   fn circle<T: Real<Scalar = f64>>(t: T, radius: f64) -> T {
///       (t * 2.0R * std::f64::consts::PI).cos() * radius
///   }
#[proc_macro_attribute]
pub fn real_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let rewritten = rewrite_tokens(quote! { #input });
    rewritten.into()
}

fn rewrite_tokens(stream: TokenStream2) -> TokenStream2 {
    stream
        .into_iter()
        .flat_map(|tt| match tt {
            TokenTree::Literal(lit) => rewrite_literal(lit),
            TokenTree::Group(group) => {
                let delim = group.delimiter();
                let inner = rewrite_tokens(group.stream());
                let new_group = proc_macro2::Group::new(delim, inner);
                vec![TokenTree::Group(new_group)]
            }
            other => vec![other],
        })
        .collect()
}

fn rewrite_literal(lit: Literal) -> Vec<TokenTree> {
    let repr = lit.to_string();

    // Strip trailing `R` suffix (e.g. "2.0R", "3R", "1_000R")
    let Some(stripped) = repr.strip_suffix('R') else {
        return vec![TokenTree::Literal(lit)];
    };

    // Validate the stripped text is actually a number.
    if stripped.parse::<f64>().is_err() {
        return vec![TokenTree::Literal(lit)];
    }

    // Re-lex the digits as an unsuffixed literal so the compiler sees a clean f64.
    let value: f64 = stripped.parse().unwrap();
    let value_lit = Literal::f64_unsuffixed(value);

    let span = Span::call_site();
    let t = proc_macro2::Ident::new("T", span);
    let from_f64 = proc_macro2::Ident::new("from_f64", span);

    let expanded = quote! { #t::#from_f64(#value_lit) };
    expanded.into_iter().collect()
}

// ── compose_params ───────────────────────────────────────────────────────────

use quote::format_ident;
use syn::parse::{Parse, ParseStream};
use syn::{braced, bracketed, Attribute, Ident, LitInt, Token, Type, Visibility};

struct ComposeParamsInput {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
    ty_param: Ident,
    fields: Vec<(Ident, Type, usize)>,
}

impl Parse for ComposeParamsInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![<]>()?;
        let ty_param: Ident = input.parse()?;
        input.parse::<Token![>]>()?;
        let body;
        braced!(body in input);
        let mut fields = Vec::new();
        while !body.is_empty() {
            let field: Ident = body.parse()?;
            body.parse::<Token![:]>()?;
            let ty: Type = body.parse()?;
            let dims;
            bracketed!(dims in body);
            let n: LitInt = dims.parse()?;
            fields.push((field, ty, n.base10_parse()?));
            if body.peek(Token![,]) {
                body.parse::<Token![,]>()?;
            }
        }
        Ok(Self { attrs, vis, name, ty_param, fields })
    }
}

/// Defines a parameter struct whose fields each convert to/from a fixed-size
/// array, generating:
///
/// - `From<[T; DIM]>` and `From<Struct> for [T; DIM]` (chunk concatenation);
/// - `Struct::<T>::DIM` — the total parameter count;
/// - one associated fn per field, `Struct::<T>::field(key) -> BlockRef<n>`,
///   mapping a problem-builder `BlockKey<DIM>` to that field's sub-block —
///   so factor wiring never repeats offset/dimension literals.
///
/// ```ignore
/// use odysseus_solver::compose_params;
/// use odysseus_solver::math3d::Vec3;
///
/// compose_params! {
///     pub struct MyParams<T> {
///         position: Vec3<T> [3],
///         velocity: Vec3<T> [3],
///     }
/// }
///
/// let arr: [f64; MyParams::<f64>::DIM] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let p = MyParams::from(arr);
/// let back: [f64; 6] = p.into();
///
/// let mut pb = odysseus_solver::problem::Problem::new();
/// let key = pb.add_block(back);
/// let vel_ref = MyParams::<f64>::velocity(key); // BlockRef<3> at offset 3
/// # let _ = (p, vel_ref);
/// ```
#[proc_macro]
pub fn compose_params(input: TokenStream) -> TokenStream {
    let ComposeParamsInput { attrs, vis, name, ty_param, fields } =
        parse_macro_input!(input as ComposeParamsInput);
    let t = &ty_param;
    let dim: usize = fields.iter().map(|(_, _, n)| n).sum();
    let dim_lit = Literal::usize_unsuffixed(dim);

    let field_names: Vec<_> = fields.iter().map(|(f, _, _)| f).collect();
    let field_types: Vec<_> = fields.iter().map(|(_, ty, _)| ty).collect();
    let field_dims: Vec<_> = fields
        .iter()
        .map(|(_, _, n)| Literal::usize_unsuffixed(*n))
        .collect();

    // Per-field offsets, computed here so the emitted code is all literals.
    let mut offset = 0usize;
    let field_offsets: Vec<_> = fields
        .iter()
        .map(|(_, _, n)| {
            let lit = Literal::usize_unsuffixed(offset);
            offset += n;
            lit
        })
        .collect();

    // Associated block-ref helper per field. Fields and associated functions
    // live in different namespaces, so the fn can reuse the field's name.
    let block_fns = fields.iter().zip(&field_offsets).map(|((f, _, n), off)| {
        let n_lit = Literal::usize_unsuffixed(*n);
        let doc = format!(
            "Sub-block view of `{f}` ({n} dims at offset {off}) within a \
             problem-builder block of this struct."
        );
        quote! {
            #[doc = #doc]
            #[allow(dead_code)]
            pub fn #f(
                key: ::odysseus_solver::problem::BlockKey<#dim_lit>,
            ) -> ::odysseus_solver::problem::BlockRef<#n_lit> {
                key.sub::<#n_lit>(#off)
            }
        }
    });

    let from_chunks = field_names.iter().zip(&field_types).zip(&field_dims).zip(&field_offsets).map(
        |(((f, ty), n), off)| {
            quote! {
                let #f = {
                    let chunk: [#t; #n] = ::std::array::from_fn(|i| arr[#off + i]);
                    <#ty>::from(chunk)
                };
            }
        },
    );
    let into_chunks = field_names.iter().zip(&field_types).zip(&field_dims).zip(&field_offsets).map(
        |(((f, _ty), n), off)| {
            quote! {
                let chunk: [#t; #n] = <[#t; #n]>::from(val.#f);
                out[#off..#off + #n].copy_from_slice(&chunk);
            }
        },
    );

    let _ = format_ident!("_unused"); // keep format_ident import warm for future use

    quote! {
        #(#attrs)*
        #vis struct #name<#t> {
            #(pub #field_names: #field_types,)*
        }

        impl<#t> #name<#t> {
            /// Total parameter count across all fields.
            pub const DIM: usize = #dim_lit;

            #(#block_fns)*
        }

        impl<#t: Copy + Default> ::std::convert::From<[#t; #dim_lit]> for #name<#t> {
            fn from(arr: [#t; #dim_lit]) -> Self {
                #(#from_chunks)*
                Self { #(#field_names,)* }
            }
        }

        impl<#t: Copy + Default> ::std::convert::From<#name<#t>> for [#t; #dim_lit] {
            fn from(val: #name<#t>) -> Self {
                let mut out = [#t::default(); #dim_lit];
                #(#into_chunks)*
                out
            }
        }
    }
    .into()
}
