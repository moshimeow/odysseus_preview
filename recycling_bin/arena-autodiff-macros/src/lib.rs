use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    Expr, Ident, Token,
};

/// Expression macro with proper operator precedence
///
/// Usage: expr!(math_ctx, expression)
///
/// The type is inferred from the MathCtx<T> parameter.
///
/// Supports:
/// - Binary operations: +, -, *, / (with correct precedence)
/// - Function calls: sin(x), cos(x), sqrt(x)
/// - Parentheses for grouping
/// - Variables and literals
///
/// ## Examples:
/// ```ignore
/// let mut arena = JetArena::new(2);
/// let mut ctx = MathCtx::<JetHandle>::new(&mut arena);
/// let result = expr!(ctx, sin(x) + y);
/// ```
#[proc_macro]
pub fn expr(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ExprInput);
    let code = generate_code(&input.ctx_expr, &input.expr);
    TokenStream::from(quote! { #code })
}

/// Parsed input: math_ctx, expression
struct ExprInput {
    ctx_expr: Expr,
    expr: MathExpr,
}

impl Parse for ExprInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse: math_ctx, expression
        let ctx_expr: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let expr = parse_expression(input)?;
        Ok(ExprInput { ctx_expr, expr })
    }
}

/// Mathematical expression AST
#[derive(Debug, Clone)]
enum MathExpr {
    Literal(syn::Lit),
    Variable(Ident),
    FieldAccess {
        base: Box<MathExpr>,
        field: Ident,
    },
    Binary {
        left: Box<MathExpr>,
        op: BinOp,
        right: Box<MathExpr>,
    },
    FunctionCall {
        name: Ident,
        arg: Box<MathExpr>,
    },
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinOp {
    fn precedence(&self) -> u8 {
        match self {
            BinOp::Add | BinOp::Sub => 1,
            BinOp::Mul | BinOp::Div => 2,
        }
    }
}

/// Parse expression with operator precedence using recursive descent
fn parse_expression(input: ParseStream) -> syn::Result<MathExpr> {
    parse_binary_expr(input, 0)
}

/// Parse binary expression with precedence climbing
fn parse_binary_expr(input: ParseStream, min_prec: u8) -> syn::Result<MathExpr> {
    let mut left = parse_unary_expr(input)?;

    loop {
        // Check if we're at the end or not looking at an operator
        if input.is_empty() {
            break;
        }

        // Try to peek for an operator
        let op = if input.peek(Token![+]) {
            BinOp::Add
        } else if input.peek(Token![-]) {
            BinOp::Sub
        } else if input.peek(Token![*]) {
            BinOp::Mul
        } else if input.peek(Token![/]) {
            BinOp::Div
        } else {
            // No operator found, we're done
            break;
        };

        // Check precedence - if this operator has lower precedence than what we're looking for, stop
        if op.precedence() < min_prec {
            break;
        }

        // Consume the operator
        match op {
            BinOp::Add => { input.parse::<Token![+]>()?; },
            BinOp::Sub => { input.parse::<Token![-]>()?; },
            BinOp::Mul => { input.parse::<Token![*]>()?; },
            BinOp::Div => { input.parse::<Token![/]>()?; },
        }

        // Parse right side with higher precedence for left-associative ops
        let right = parse_binary_expr(input, op.precedence() + 1)?;

        left = MathExpr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        };
    }

    Ok(left)
}

/// Parse unary expression (atom or function call)
fn parse_unary_expr(input: ParseStream) -> syn::Result<MathExpr> {
    // Parse the base atom
    let mut expr = parse_atom(input)?;

    // Handle field access (e.g., rvec.x, rvec.y)
    while input.peek(Token![.]) {
        input.parse::<Token![.]>()?;
        let field: Ident = input.parse()?;
        expr = MathExpr::FieldAccess {
            base: Box::new(expr),
            field,
        };
    }

    Ok(expr)
}

/// Parse atomic expression (literal, variable, function call, or parenthesized)
fn parse_atom(input: ParseStream) -> syn::Result<MathExpr> {
    // Try to parse as function call first
    if input.peek(Ident) && input.peek2(syn::token::Paren) {
        let func_name: Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        let arg = parse_expression(&content)?;
        return Ok(MathExpr::FunctionCall {
            name: func_name,
            arg: Box::new(arg),
        });
    }

    // Try to parse as parenthesized expression
    if input.peek(syn::token::Paren) {
        let content;
        syn::parenthesized!(content in input);
        return parse_expression(&content);
    }

    // Try to parse as literal
    if input.peek(syn::Lit) {
        let lit: syn::Lit = input.parse()?;
        return Ok(MathExpr::Literal(lit));
    }

    // Parse as variable
    let ident: Ident = input.parse()?;
    Ok(MathExpr::Variable(ident))
}

/// Generate Rust code from the AST
///
/// Generates inline code that reborrows the context for each operation
fn generate_code(ctx_expr: &Expr, expr: &MathExpr) -> proc_macro2::TokenStream {
    generate_code_with_ctx(ctx_expr, expr)
}

/// Internal code generation that passes context through and reborrows at each operation
fn generate_code_with_ctx(ctx_expr: &Expr, expr: &MathExpr) -> proc_macro2::TokenStream {
    match expr {
        MathExpr::Literal(lit) => {
            quote! { (#ctx_expr).constant(#lit) }
        }
        MathExpr::Variable(var) => {
            quote! { #var }
        }
        MathExpr::FieldAccess { base, field } => {
            let base_code = generate_code_with_ctx(ctx_expr, base);
            quote! { (#base_code).#field }
        }
        MathExpr::Binary { left, op, right } => {
            let left_code = generate_code_with_ctx(ctx_expr, left);
            let right_code = generate_code_with_ctx(ctx_expr, right);
            let method = match op {
                BinOp::Add => quote! { add },
                BinOp::Sub => quote! { sub },
                BinOp::Mul => quote! { mul },
                BinOp::Div => quote! { div },
            };
            // Use temporaries to ensure sequential evaluation and avoid overlapping borrows
            quote! {
                {
                    let _tmp_left = #left_code;
                    let _tmp_right = #right_code;
                    (#ctx_expr).#method(_tmp_left, _tmp_right)
                }
            }
        }
        MathExpr::FunctionCall { name, arg } => {
            let arg_code = generate_code_with_ctx(ctx_expr, arg);
            let func_name = name.to_string();
            let method = match func_name.as_str() {
                "sin" => quote! { sin },
                "cos" => quote! { cos },
                "sqrt" => quote! { sqrt },
                _ => panic!("Unsupported function: {}", func_name),
            };
            // Use temporary to ensure argument completes before function call
            quote! {
                {
                    let _tmp_arg = #arg_code;
                    (#ctx_expr).#method(_tmp_arg)
                }
            }
        }
    }
}
