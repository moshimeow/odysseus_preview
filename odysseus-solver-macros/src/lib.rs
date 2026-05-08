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
