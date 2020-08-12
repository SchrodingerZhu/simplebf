#include <tao/pegtl.hpp>
#include <tao/pegtl/contrib/parse_tree.hpp>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/Verifier.h>
#include <cxxopts.hpp>
#include <iostream>

namespace grammar {
    using namespace tao::pegtl;
    struct ShiftR : one<'>'> {};
    struct ShiftL : one<'<'> {};
    struct Inc : one<'+'> {};
    struct Dec : one<'-'> {};
    struct Dot : one<'.'> {};
    struct Comma : one<','> {};
    struct Loop;
    struct NonSense : plus<not_one<'<', '>', '+', '-', '.', ',', '[', ']'>> {};
    struct Statement : plus<sor<ShiftR, ShiftL, Inc, Dec, Dot, Comma, Loop, NonSense>> {};
    struct Loop : seq<one<'['>, Statement, one<']'>> {};
    struct Module : must<Statement> {};
}

namespace ast {
    using namespace llvm;
    struct ASTNode {
        static LLVMContext context;
        static std::unique_ptr<Module> module;
        static IRBuilder<> builder;
        static Function* toplevel;
        static Function* getchar;
        static Function* putchar;
        static AllocaInst* pointer;
        static AllocaInst* array;

        virtual Value* codegen() = 0;

    };

    LLVMContext ASTNode::context {};
    IRBuilder<> ASTNode::builder {context};
    std::unique_ptr<Module> ASTNode::module = std::make_unique<Module>("simple", context);
    Function* ASTNode::toplevel;
    Function* ASTNode::getchar;
    Function* ASTNode::putchar;
    AllocaInst* ASTNode::pointer;
    AllocaInst* ASTNode::array;

    struct Statement : ASTNode {
        std::vector<std::unique_ptr<ASTNode>> children;
        Statement(std::vector<std::unique_ptr<ASTNode>>&& children) : children(std::move(children)) {}
        Value* codegen() override {
            Value* end = nullptr;
            for (auto& i: children) {
                end = i->codegen();
            }
            return end;
        }
    };

    struct ShiftR : ASTNode {
        Value* codegen() override {
            auto cur = builder.CreateLoad(pointer);
            auto one  = ConstantInt::get( context , APInt(32, StringRef("1"), 10));
            auto next = builder.CreateAdd(cur, one);
            return builder.CreateStore(next, pointer);
        }
    };

    struct ShiftL : ASTNode {
        Value* codegen() override {
            auto cur = builder.CreateLoad(pointer);
            auto one  = ConstantInt::get( context , APInt(32, StringRef("1"), 10));
            auto next = builder.CreateSub(cur, one);
            return builder.CreateStore(next, pointer);
        }
    };

    struct Dot : ASTNode {
        Value* codegen() override {
            auto index = builder.CreateLoad(pointer);
            auto cur_ptr = builder.CreateGEP(array, index);
            auto ele = builder.CreateLoad(cur_ptr);
            return builder.CreateCall(putchar, {ele});
        }
    };

    struct Comma : ASTNode {
        Value* codegen() override {
            auto cur = builder.CreateLoad(pointer);
            auto input = builder.CreateCall(getchar, {});
            auto ptr = builder.CreateGEP(array, cur);
            return builder.CreateStore(input, ptr);
        }
    };

    struct Inc : ASTNode {
        Value* codegen() override {
            auto cur = builder.CreateLoad(pointer);
            auto ptr = builder.CreateGEP(array, cur);
            auto val = builder.CreateLoad(ptr);
            auto one  = ConstantInt::get( context , APInt(32, StringRef("1"), 10));
            auto next = builder.CreateAdd(val, one);
            return builder.CreateStore(next, ptr);
        }
    };

    struct Dec : ASTNode {
        Value* codegen() override {
            auto cur = builder.CreateLoad(pointer);
            auto ptr = builder.CreateGEP(array, cur);
            auto val = builder.CreateLoad(ptr);
            auto one  = ConstantInt::get( context , APInt(32, StringRef("1"), 10));
            auto next = builder.CreateSub(val, one);
            return builder.CreateStore(next, ptr);
        }
    };

    struct Loop : ASTNode {
        std::unique_ptr<ASTNode> child;
        explicit Loop(std::unique_ptr<ASTNode> child) : child(std::move(child)) {}
        Value* codegen() override {
            auto loop = BasicBlock::Create(context, "", toplevel);
            auto after = BasicBlock::Create(context, "", toplevel);
            auto zero  = ConstantInt::get( context , APInt(32, StringRef("0"), 10));
            {
                auto index = builder.CreateLoad(pointer);
                auto cur_ptr = builder.CreateGEP(array, index);
                auto cur = builder.CreateLoad(cur_ptr);
                auto flag = builder.CreateICmpEQ(cur, zero);
                builder.CreateCondBr(flag, after, loop);
            }
            builder.SetInsertPoint(loop);
            if(child) child->codegen();
            {
                auto index = builder.CreateLoad(pointer);
                auto cur_ptr = builder.CreateGEP(array, index);
                auto cur = builder.CreateLoad(cur_ptr);
                auto flag = builder.CreateICmpEQ(cur, zero);
                builder.CreateCondBr(flag, after, loop);
            }
            builder.SetInsertPoint(after);
            return after;
        }
    };

    struct Module : ASTNode {
        std::unique_ptr<ASTNode> child;
        explicit Module(std::unique_ptr<ASTNode> child) : child(std::move(child)) {}
        Value* codegen() override {
            auto int_type = IntegerType::get(context, 32);
            {
                auto func_type = FunctionType::get(int_type, {int_type}, false);
                putchar = Function::Create(func_type, Function::ExternalLinkage, "putchar", module.get());
            }
            {
                auto func_type = FunctionType::get(int_type, {}, false);
                getchar = Function::Create(func_type, Function::ExternalLinkage, "getchar", module.get());
            }
            auto func_type = FunctionType::get(int_type, {}, false);
            toplevel = Function::Create(func_type, Function::ExternalLinkage, "main", module.get());
            auto basic_block = BasicBlock::Create(context, "", toplevel);
            builder.SetInsertPoint(basic_block);
            array = builder.CreateAlloca(int_type, ConstantInt::get( context , APInt(32, StringRef("10000"), 10)));
            pointer = builder.CreateAlloca(int_type, nullptr);
            auto zero  = ConstantInt::get( context , APInt(32, StringRef("0"), 10));
            builder.CreateStore(zero, pointer);
            if(child) child->codegen();
            builder.CreateRet(zero);
            return toplevel;
        }
    };

}
static int a = 0;
namespace transform {

    template <class T>
    struct selector : std::false_type {};

#define select(X) \
    template <> \
    struct selector<grammar::X> : std::true_type {}

    select(Dot);
    select(Comma);
    select(Inc);
    select(Dec);
    select(ShiftL);
    select(ShiftR);
    select(Statement);
    select(Loop);
    select(Module);

#define trans(type, block) \
    if (n->is_type<grammar::type>()) block
#define trans_simple(type) trans(type, {return std::make_unique<ast::type> (); })

    using node = tao::pegtl::parse_tree::node;


    std::unique_ptr<ast::ASTNode> transform(const std::unique_ptr<node>& n) {

        trans(Module, {
            auto stmt = transform(n->children[0]);
            return std::make_unique<ast::Module>(std::move(stmt));
        })

        trans(Statement, {
            std::vector<std::unique_ptr<ast::ASTNode>> children;
            for (const auto& i : n->children) {
                children.emplace_back(transform(i));
            }
            return std::make_unique<ast::Statement>(std::move(children));
        })

        trans(Loop, {
            auto stmt = transform(n->children[0]);
            return std::make_unique<ast::Loop>(std::move(stmt));
        })

        trans_simple(Dot)
        trans_simple(Comma)
        trans_simple(Inc)
        trans_simple(Dec)
        trans_simple(ShiftR)
        trans_simple(ShiftL)
        __builtin_unreachable();
    }
}



int main(int argc, const char** argv) {
    cxxopts::Options options("simple brainfuck compiler", "compile brainfuck code to ELF");
    options.add_options()
            ("O,opt", "Enable optimization", cxxopts::value<bool>()->default_value("false"))
            ("i,input", "Input file name", cxxopts::value<std::string>())
            ("o,output", "Output file name", cxxopts::value<std::string>())
            ("v,verbose", "Display LLVM IR", cxxopts::value<bool>()->default_value("false"));
    auto result = options.parse(argc, argv);
    auto in = tao::pegtl::file_input(result["input"].as<std::string>());
    auto res  = tao::pegtl::parse_tree::parse<grammar::Module, transform::selector>(in);
    auto ast = transform::transform(res->children[0]);
    ast->codegen();
    if (llvm::verifyFunction(*ast::ASTNode::toplevel, &llvm::errs())) {
        llvm::errs() << "failed to verify toplevel";
        return 1;
    };
    if (result["opt"].as<bool>()) {
        using namespace llvm;
        auto FPM = std::make_unique<legacy::FunctionPassManager>(ast::ASTNode::module.get());

        // Do simple "peephole" optimizations and bit-twiddling optzns.
        FPM->add(createInstructionCombiningPass());
        // Reassociate expressions.
        FPM->add(createReassociatePass());
        // Reassociate expressions.
        FPM->add(createNewGVNPass());
        // Simplify the control flow graph (deleting unreachable blocks, etc).
        FPM->add(createCFGSimplificationPass());

        FPM->doInitialization();
        FPM->run(*ast::ASTNode::toplevel);
        FPM->doFinalization();
    }
    if (result["verbose"].as<bool>()) {
        ast::ASTNode::module->print(llvm::outs(), nullptr);
    }
    auto target_triple = llvm::sys::getDefaultTargetTriple();

    std::string Error;
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    auto target = llvm::TargetRegistry::lookupTarget(target_triple, Error);

    if (!target) {
        llvm::errs() << Error;
        return 1;
    }

    auto CPU = "generic";
    auto Features = "";

    llvm::TargetOptions opt;
    auto RM = llvm::Optional<llvm::Reloc::Model>();
    auto machine = target->createTargetMachine(target_triple, CPU, Features, opt, RM);
    ast::ASTNode::module->setDataLayout(machine->createDataLayout());
    ast::ASTNode::module->setTargetTriple(target_triple);
    auto filename = result["output"].as<std::string>();
    std::error_code EC;
    llvm::raw_fd_ostream dest(filename, EC);

    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message();
        return 1;
    }

    llvm::legacy::PassManager pass;
    auto file_type = llvm::CGFT_ObjectFile;

    if (machine->addPassesToEmitFile(pass, dest, nullptr, file_type)) {
        llvm::errs() << "TargetMachine can't emit a file of this type";
        return 1;
    }

    pass.run(*ast::ASTNode::module);
    dest.flush();

    llvm::outs() << "written to file\n";
    return 0;
}
