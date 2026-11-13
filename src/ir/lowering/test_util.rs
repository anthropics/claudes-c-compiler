//! Test support for the lowering phase: turn a C snippet into an `IrModule`
//! without going through the driver.

use crate::backend::Target;
use crate::common::error::DiagnosticEngine;
use crate::common::source::SourceManager;
use crate::frontend::lexer::Lexer;
use crate::frontend::parser::Parser;
use crate::frontend::sema::SemanticAnalyzer;
use crate::ir::module::IrModule;

use super::Lowerer;

/// Lex, parse, analyze and lower `source` for `target`.
///
/// `source` must already be preprocessed (no directives). Panics if the
/// snippet does not parse, fails semantic analysis or reports an error during
/// lowering, so that a test cannot pass on a snippet the compiler rejects.
pub(crate) fn lower_source(target: Target, source: &str) -> IrModule {
    // The same per-thread target setup the driver does before compiling.
    crate::common::types::set_target_ptr_size(target.ptr_size());
    crate::common::types::set_target_long_double_is_f128(
        matches!(target, Target::Aarch64 | Target::Riscv64)
    );

    let mut source_manager = SourceManager::new();
    let file_id = source_manager.add_file("test.c".to_string(), source.to_string());
    source_manager.build_line_map();
    let tokens = Lexer::new(source_manager.get_content(file_id), file_id).tokenize();

    let mut diagnostics = DiagnosticEngine::new();
    diagnostics.set_source_manager(source_manager);
    let mut parser = Parser::new(tokens);
    parser.set_diagnostics(diagnostics);
    let ast = parser.parse();
    assert_eq!(parser.error_count, 0, "test source does not parse");

    let mut sema = SemanticAnalyzer::new();
    sema.set_diagnostics(parser.take_diagnostics());
    assert!(sema.analyze(&ast).is_ok(), "test source fails semantic analysis");
    let diagnostics = sema.take_diagnostics();
    let sema_result = sema.into_result();

    let lowerer = Lowerer::with_type_context(
        target,
        sema_result.type_context,
        sema_result.functions,
        sema_result.expr_types,
        sema_result.const_values,
        diagnostics,
        false,
    );
    let (module, diagnostics) = lowerer.lower(&ast);
    assert!(!diagnostics.has_errors(), "test source fails IR lowering");
    module
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::module::GlobalInit;

    #[test]
    fn test_lower_source_returns_globals_and_functions() {
        let module = lower_source(Target::X86_64, "int answer = 42;\nint get(void) { return answer; }\n");

        let answer = module.globals.iter().find(|g| g.name == "answer").expect("global `answer`");
        assert!(matches!(answer.init, GlobalInit::Scalar(_)));
        assert!(module.functions.iter().any(|f| f.name == "get"));
    }
}
