/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <cassert>
#include "trace/hook/HookHandlers.h"
#include "trace/type/RecordingDataStructure.h"
#include "trace/hook/HookContext.h"
#include "analyse/PieChartAnalyzer.h"

extern "C" {
#define setbit(x, y) x|=(1<<y)
#define clrbit(x, y) x&=～(1<<y)
#define chkbit(x, y) x&(1<<y)


#define PUSHZMM(ArgumentName) \
"subq $64,%rsp\n\t" \
"vmovdqu64  %zmm"#ArgumentName" ,(%rsp)\n\t"

#define POPZMM(ArgumentName) \
"vmovdqu64  (%rsp),%zmm"#ArgumentName"\n\t"\
"addq $64,%rsp\n\t"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

/**
* Restore Registers
*/


#define SAVE_COMPACT
#ifdef SAVE_ALL
#define SAVE_PRE  \
    /*The stack should be 16 bytes aligned before start this block*/ \
    /*ZMM registers are ignored. Normally we do not use them in out hook*/ \
    /*Parameter passing registers*/       \
    "movq %rax,0x08(%rsp)\n\t" /*8 bytes*/\
    "movq %rcx,0x10(%rsp)\n\t" /*8 bytes*/\
    "movq %rdx,0x18(%rsp)\n\t" /*8 bytes*/\
    "movq %rsi,0x20(%rsp)\n\t" /*8 bytes*/\
    "movq %rdi,0x28(%rsp)\n\t" /*8 bytes*/\
    "movq %r8,0x30(%rsp)\n\t"  /*8 bytes*/\
    "movq %r9,0x38(%rsp)\n\t"  /*8 bytes*/\
    "movq %r10,0x40(%rsp)\n\t" /*8 bytes*/\
    /*Call-ee saved registers */ \
    "movq %rbx,0x48(%rsp)\n\t" /*8 bytes*/\
    "movq %rbp,0x50(%rsp)\n\t" /*8 bytes*/\
    "movq %r12,0x58(%rsp)\n\t" /*8 bytes*/\
    "movq %r13,0x60(%rsp)\n\t" /*8 bytes*/\
    "movq %r14,0x68(%rsp)\n\t" /*8 bytes*/\
    "movq %r15,0x70(%rsp)\n\t" /*8 bytes*/\
    "stmxcsr 0x78(%rsp)\n\t" /*16bytes*/  \
    "fstcw 0xEA(%rsp)\n\t"                \
    /*The following addr must be 16 bits aligned*/ \
    /*Save XMM1 to XMM7*/                          \
    "vmovdqu64 %zmm0,0x88(%rsp) \n\t"/*64bytes*/  \
    "vmovdqu64 %zmm1,0xC8(%rsp) \n\t"/*64bytes*/  \
    "vmovdqu64 %zmm2,0x108(%rsp) \n\t"/*64bytes*/  \
    "vmovdqu64 %zmm3,0x148(%rsp) \n\t"/*64bytes*/  \
    "vmovdqu64 %zmm4,0x188(%rsp) \n\t"/*64bytes*/  \
    "vmovdqu64 %zmm5,0x1C8(%rsp) \n\t"/*64bytes*/  \
    "vmovdqu64 %zmm6,0x208(%rsp) \n\t"/*64bytes*/  \
    "vmovdqu64 %zmm7,0x248(%rsp) \n\t"/*64bytes*/  \

#define SAVE_BYTES_PRE "0x288" //0x248+512
#define SAVE_BYTES_PRE_plus8 "0x290" //0x288+0x8
#define SAVE_BYTES_PRE_plus16 "0x298" //0x288+0x8

//Do not write restore by yourself, copy previous code and reverse operand order
#define RESTORE_PRE  \
    "movq 0x08(%rsp),%rax\n\t" /*8 bytes*/\
    "movq 0x10(%rsp),%rcx\n\t" /*8 bytes*/\
    "movq 0x18(%rsp),%rdx\n\t" /*8 bytes*/\
    "movq 0x20(%rsp),%rsi\n\t" /*8 bytes*/\
    "movq 0x28(%rsp),%rdi\n\t" /*8 bytes*/\
    "movq 0x30(%rsp),%r8\n\t"  /*8 bytes*/\
    "movq 0x38(%rsp),%r9\n\t"  /*8 bytes*/\
    "movq 0x40(%rsp),%r10\n\t" /*8 bytes*/\
    /*Call-ee saved registers */ \
    "movq 0x48(%rsp),%rbx\n\t" /*8 bytes*/\
    "movq 0x50(%rsp),%rbp\n\t" /*8 bytes*/\
    "movq 0x58(%rsp),%r12\n\t" /*8 bytes*/\
    "movq 0x60(%rsp),%r13\n\t" /*8 bytes*/\
    "movq 0x68(%rsp),%r14\n\t" /*8 bytes*/\
    "movq 0x70(%rsp),%r15\n\t" /*8 bytes*/\
    "ldmxcsr 0x78(%rsp)\n\t" /*16bytes*/  \
    "fldcw 0xEA(%rsp)\n\t"                \
    /*The following addr must be 16 bits aligned*/ \
    /*Save XMM1 to XMM7*/                          \
    "vmovdqu64 0x88(%rsp),%zmm0 \n\t"/*512bytes*/  \
    "vmovdqu64 0xC8(%rsp),%zmm1 \n\t"/*512bytes*/  \
    "vmovdqu64 0x108(%rsp),%zmm2 \n\t"/*512bytes*/  \
    "vmovdqu64 0x148(%rsp),%zmm3 \n\t"/*512bytes*/  \
    "vmovdqu64 0x188(%rsp),%zmm4 \n\t"/*512bytes*/  \
    "vmovdqu64 0x1C8(%rsp),%zmm5 \n\t"/*512bytes*/  \
    "vmovdqu64 0x208(%rsp),%zmm6 \n\t"/*512bytes*/  \
    "vmovdqu64 0x248(%rsp),%zmm7 \n\t"/*512bytes*/  \


#define SAVE_POST  \
    /*The stack should be 16 bytes aligned before start this block*/       \
    /*ZMM registers are ignored. Normally we do not use them in out hook*/ \
    /*Parameter passing registers*/                                        \
    "movq %rax,(%rsp)\n\t" /*8 bytes*/                                     \
    "movq %rdx,0x8(%rsp)\n\t" /*8 bytes*/                                  \
    "vmovdqu64 %zmm0,0x10(%rsp) \n\t"/*64bytes*/                           \
    "vmovdqu64 %zmm1,0x50(%rsp) \n\t"/*64bytes*/                           \
    /*https://www.cs.mcgill.ca/~cs573/winter2001/AttLinux_syntax.htm*/     \
    "fsave 0x90(%rsp)\n\t" /*108bytes*/                                              \

#define SAVE_BYTES_POST "0xFC" /*0x90+108*/


#define RESTORE_POST  \
    /*Parameter passing registers*/                                        \
    "movq (%rsp),%rax\n\t" /*8 bytes*/                                     \
    "movq 0x8(%rsp),%rdx\n\t" /*8 bytes*/                                  \
    "vmovdqu64 0x10(%rsp),%zmm0 \n\t"/*64bytes*/                           \
    "vmovdqu64 0x50(%rsp),%zmm1 \n\t"/*64bytes*/                           \
    /*https://www.cs.mcgill.ca/~cs573/winter2001/AttLinux_syntax.htm*/     \
    "fnsave 0x90(%rsp)\n\t" /*108bytes*/
#endif

//In this mode, we reduce registers that is not used.
#ifdef SAVE_COMPACT
#define SAVE_PRE  \
    /*The stack should be 16 bytes aligned before start this block*/ \
    /*ZMM registers are ignored. Normally we do not use them in out hook*/ \
    /*Parameter passing registers*/       \
    "subq $" SAVE_BYTES_PRE ",%rsp\n\t" /*rsp -= SAVE_BYTES_PRE*/ \
    "movq %rax,0x08(%rsp)\n\t" /*8 bytes*/\
    "movq %rcx,0x10(%rsp)\n\t" /*8 bytes*/\
    "movq %rdx,0x18(%rsp)\n\t" /*8 bytes*/\
    "movq %rsi,0x20(%rsp)\n\t" /*8 bytes*/\
    "movq %rdi,0x28(%rsp)\n\t" /*8 bytes*/\
    "movq %r8,0x30(%rsp)\n\t"  /*8 bytes*/\
    "movq %r9,0x38(%rsp)\n\t"  /*8 bytes*/\
    "movq %r10,0x40(%rsp)\n\t" /*8 bytes*/\

#define SAVE_BYTES_PRE_plus8 "0x60"
#define SAVE_BYTES_PRE "0x58" //0x40+0x10+0x8 (Funcid) 16 bit aligned.
#define SAVE_BYTES_PRE_minus8 "0x50" //0x50+0x8 (GotAddr)
#define SAVE_BYTES_PRE_minus16 "0x48" //0x50+0x10 (LoadingId)
#define SAVE_BYTES_PRE_minus24 "0x40" //0x50+0x18 (funcId)

//Do not write restore by yourself, copy previous code and reverse operand order
#define RESTORE_PRE  \
    "movq 0x08(%rsp),%rax\n\t" /*8 bytes*/\
    "movq 0x10(%rsp),%rcx\n\t" /*8 bytes*/\
    "movq 0x18(%rsp),%rdx\n\t" /*8 bytes*/\
    "movq 0x20(%rsp),%rsi\n\t" /*8 bytes*/\
    "movq 0x28(%rsp),%rdi\n\t" /*8 bytes*/\
    "movq 0x30(%rsp),%r8\n\t"  /*8 bytes*/\
    "movq 0x38(%rsp),%r9\n\t"  /*8 bytes*/\
    "movq 0x40(%rsp),%r10\n\t" /*8 bytes*/\

#define RESTORE_PRE_OVERRIDE_RETURN \
    RESTORE_PRE \
    "addq $" SAVE_BYTES_PRE_plus8 ",%rsp\n\t" //Plus 8 to override call addr

#define RESTORE_PRE_NO_OVERRIDE_RETURN \
    RESTORE_PRE \
    "addq $" SAVE_BYTES_PRE ",%rsp\n\t" //Plus 16 is because there are two push to save id


#define SAVE_POST  \
    /*The stack should be 16 bytes aligned before start this block*/       \
    /*ZMM registers are ignored. Normally we do not use them in out hook*/ \
    /*Parameter passing registers*/                                        \
    "subq $" SAVE_BYTES_POST ",%rsp\n\t"                                   \
    "movq %rax,(%rsp)\n\t" /*8 bytes*/                                     \
    "movq %rdx,0x8(%rsp)\n\t" /*8 bytes*/                                  \
    "movdqu %xmm0,0x10(%rsp) \n\t"/*16bytes*/                              \
    "movdqu %xmm1,0x20(%rsp) \n\t"/*16bytes*/
/*https://www.cs.mcgill.ca/~cs573/winter2001/AttLinux_syntax.htm*/
/*"fsave 0x10(%rsp)\n\t"*/ /*108bytes*/

#define SAVE_BYTES_POST "0x30" /*0x20+18*/


#define RESTORE_POST  \
    /*Parameter passing registers*/                                        \
    "movq (%rsp),%rax\n\t" /*8 bytes*/                                     \
    "movq 0x8(%rsp),%rdx\n\t" /*8 bytes*/                                  \
    "movdqu 0x10(%rsp),%xmm0 \n\t"/*64bytes*/                           \
    "movdqu 0x20(%rsp),%xmm1 \n\t"/*64bytes*/                           \
    /*https://www.cs.mcgill.ca/~cs573/winter2001/AttLinux_syntax.htm*/     \
    "addq $" SAVE_BYTES_POST ",%rsp\n\t" /*Plus 8 is because there was a push to save 8 bytes more funcId. Another 8 is to replace return address*/ \
    /*"fnsave 0x10(%rsp)\n\t"*/ /*108bytes*/
#endif


/**
 * Source code version for #define IMPL_ASMHANDLER
 * We can't add comments to a macro
 */
void __attribute__((naked)) asmTimingHandler() {
    //todo: Calculate values based on rsp rathe than saving them to registers
    __asm__ __volatile__ (
        /**
        * Save the environment, mostly
        */
            SAVE_PRE

            /**
             * Getting PLT entry address and caller address from stack
             */
            "movq " SAVE_BYTES_PRE"(%rsp),%rdi\n\t" //First parameter, return addr
            "movq " SAVE_BYTES_PRE_minus8 "(%rsp),%rsi\n\t" //Second parameter, symbolId (Pushed to stack by idsaver)
            "movq " SAVE_BYTES_PRE_minus16 "(%rsp),%rdx\n\t" //Third parameter, GOTAddr (Pushed to stack by idsaver)

            /**
             * Pre-Hook
             */
            "call preHookHandler@plt\n\t"

            // preHookHandler will return rax, and rdi here. 
            //Save return value to R11. This is the address of real function parsed by handler.
            //The return address is maybe the real function address. Or a pointer to the pseodoPlt table
            "movq %rax,%r11\n\t"
            "cmpq $1234,%rdi\n\t"
            "jz  RET_PREHOOK_ONLY\n\t"

            //=======================================> if rdi!=$1234
            /**
             * Call actual function
             */
            RESTORE_PRE_OVERRIDE_RETURN
            "callq *%r11\n\t"

            /**
             * Call after hook
             */
            //Save return value to stack
            SAVE_POST

            /**
             * Call After Hook
             */
            //todo: This line has compilation error on the server
            "call postHookHandler@plt\n\t"
            //Save return value to R11. R11 now has the address of caller.
            "movq %rax,%r11\n\t"

            /**
            * Restore return value
            */
            RESTORE_POST
            //Retrun to caller
            "jmpq *%r11\n\t"


            //=======================================> if rdi==$1234
            "RET_PREHOOK_ONLY:\n\t"
            RESTORE_PRE_NO_OVERRIDE_RETURN
            //Restore rsp to original value (Uncomment the following to only enable prehook)
            "jmpq *%r11\n\t"

            );

}


uint8_t *ldCallers = nullptr;

//Return magic number definition:
//1234: address resolved, pre-hook only (Fastest)
//else pre+afterhook. Check hook installation in afterhook
__attribute__((used)) void *preHookHandler(uint64_t nextCallAddr, int64_t symId, void **realAddrPtr) {
    using namespace mlinsight;
    assert(curContext!=nullptr);
    HookContext *curContextPtr = curContext;

    //Assumes _this->allExtSymbol won't change

    /**
    * No counting, no measuring time (If mlinsight is not installed, then tls is not initialized)
    * This may happen for external function call before the initTLS in dummy thread function
    */
    if (unlikely(bypassCHooks == MLINSIGHT_TRUE)) {
        //Skip afterhook
        asm volatile ("movq $1234, %%rdi" : /* No outputs. */
                :/* No inputs. */:"rdi");
        return *realAddrPtr;
    } else if (unlikely(curContextPtr->indexPosi >= MAX_CALL_DEPTH)) {
        //Skip afterhook
        asm volatile ("movq $1234, %%rdi" : /* No outputs. */
                :/* No inputs. */:"rdi");
        return *realAddrPtr;
    } else if (unlikely((uint64_t) curContextPtr->hookTuple[curContextPtr->indexPosi].callerAddr == nextCallAddr)) {
        //Skip afterhook
        asm volatile ("movq $1234, %%rdi" : /* No outputs. */
                :/* No inputs. */:"rdi");
        return *realAddrPtr;
    }

    bypassCHooks = MLINSIGHT_TRUE;
    //DANGEROUS: Debug Only. Contention may occur for multithreaded dynamic loading.
    //In production, we should not access curELFInfoMap and curSymInfoMap
    //mlinsight::ExtSymInfo &curElfSymInfo = curContextPtr->_this->allExtSymbol.internalArray[loadingId].internalArray[symId];

    //INFO_LOG("PreHookHandler");

    //INFO_LOGS("[Pre Hook] Thread:%lu LoadingId:%ld Func:%ld RetAddr:%p Timestamp: %lu\n", pthread_self(),loadingId, symId, (void *) nextCallAddr, getunixtimestampms());
    //assert(curContext != nullptr);ls


    /**
    * Setup environment for afterhook
    */
    ++curContextPtr->indexPosi;

    curContextPtr->hookTuple[curContextPtr->indexPosi].id.symId = symId;
    curContextPtr->hookTuple[curContextPtr->indexPosi].callerAddr = nextCallAddr;

    mlinsight::preHookAttribution(curContextPtr);

    bypassCHooks = MLINSIGHT_FALSE;
    return *realAddrPtr;
}


void *postHookHandler() {
    using namespace mlinsight;
    bypassCHooks = MLINSIGHT_TRUE;
    HookContext *curContextPtr = curContext;
    assert(curContext != nullptr);

    mlinsight::SymID symbolId = curContextPtr->hookTuple[curContextPtr->indexPosi].id.symId;
    void *callerAddr = (void *) curContextPtr->hookTuple[curContextPtr->indexPosi].callerAddr;


    if (symbolId >= curContextPtr->recordArray.getSize()) {
        curContextPtr->recordArray.allocateArray(symbolId + 1 - curContextPtr->recordArray.getSize());
    }

    int64_t &c = curContextPtr->recordArray.internalArray[symbolId].count;


    if (symbolId >= curContextPtr->recordArray.getSize()) {
        curContextPtr->recordArray.allocateArray(symbolId + 1 - curContextPtr->recordArray.getSize());
    }

    mlinsight::postHookAttribution(curContextPtr);
    //INFO_LOGS("[Post Hook] Thread:%lu Func:%ld Timestamp: %lu Duration: %lu\n", pthread_self(),symbolId, getunixtimestampms(),clockCyclesDuration);

//    INFO_LOGS("curLoadingId==%zd curContextPtr->recordArray->getSize()==%zd",curLoadingId,curContextPtr->recordArray->getSize());

//    INFO_LOGS("API duration = %lu - %lslu=%lu", postLogicalClockCycle, preLogicalClockCycle, clockCyclesDuration);


//    DBG_LOGS("Thread=%lu AttributingAPITime (%lu - %lu) / %u=%ld", pthread_self(), wallClockSnapshot,
//             preLogicalClockCycle,
//             threadNumPhase, clockCyclesDuration / threadNumPhase);
//    DBG_LOGS("Thread=%lu AttributingThreadEndTime+=(%lu - %lu) / %u=%lu", pthread_self(), wallClockSnapshot,
//             curContextPtr->prevWallClockSnapshot, threadNumPhase,
//             (wallClockSnapshot - curContextPtr->prevWallClockSnapshot) / threadNumPhase);
    --curContextPtr->indexPosi;
    assert(curContextPtr->indexPosi >= 1);
    bypassCHooks = MLINSIGHT_FALSE;
    return callerAddr;
}

void __attribute__((used, naked, noinline)) myPltEntry() {
    __asm__ __volatile__ (
            "movq $0x1122334455667788,%r11\n\t"
            "jmpq *%r11\n\t"
            );
}

void __attribute__((used, naked, noinline)) callIdSaverScheme3() {
    __asm__ __volatile__ (
        /**
         * Access TLS, make sure it's initialized
         */
            "mov $0x1122334455667788,%r11\n\t"//Move the tls offset of context to r11
            "mov %fs:(%r11),%r11\n\t" //Now r11 points to the tls header
            //Check whether the context is initialized
            "cmpq $0,%r11\n\t"
            //Skip processing if context is not initialized
            "jz SKIP\n\t"

            "pushq %r10\n\t"

            "movq 0x11223344(%r11),%r11\n\t" //Fetch recordArray.internalArray's address  -> r11
            "movq 0x11223344(%r11),%r10\n\t" //Fetch recordArray.internalArray[symId].count -> r10
            "addq $1,%r10\n\t" //count + 1
            "movq %r10,0x11223344(%r11)\n\t" //Store count to recordArray[loadingId].internalArray[symId].count
            "movq 0x11223344(%r11),%r11\n\t" //Fetch recordArray[loadingId].internalArray[symId].gap -> r11
            "andq %r11,%r10\n\t" //count value (r10) % gap (r11) -> r11, gap value must be a power of 2
            "cmpq $0,%r10\n\t" //If count % gap == 0. Use timing
            "pop %r10\n\t"
            "jz TIMING_JUMPER\n\t" //Check if context is initialized

            /**
            * Return
            */
            "SKIP:"
            "movq $0x1122334455667788,%r11\n\t" //GOT address
            "jmpq *(%r11)\n\t"
            "pushq $0x11223344\n\t" //Plt stub id
            "movq $0x1122334455667788,%r11\n\t" //First plt entry
            "jmpq *%r11\n\t"

            /**
             * Timing
             */
            //Perform timing
            "TIMING_JUMPER:"
            "movl $0x44332211,-0x10(%rsp)\n\t" //Save low bits of gotEntrtAddress
            "movl $0x44332211,-0xC(%rsp)\n\t" //Save hi bits of gotEntrtAddress
            "movq $0x44332211,-0x8(%rsp)\n\t" //Save funcId to stack
            "movq $0x1122334455667788,%r11\n\t"
            "jmpq *%r11\n\t"
            );
}

void __attribute__((used, naked, noinline)) callIdSaverScheme4() {
    __asm__ __volatile__ (
            /**
             * Access TLS, make sure it's initialized
             */
            "movq $0x1122334455667788,%rcx\n\t" //Move the pointer of the original function to rdx (4th parameter)
            "movq $0x1122334455667788,%r11\n\t" //Jump to the generalPyHookFunction
            "jmpq *%r11\n\t"
            );
}

void __attribute__((used, naked, noinline)) callIdSaverScheme5() {
    __asm__ __volatile__ (
            /**
             * Access TLS, make sure it's initialized
             */
            "movq $0x1122334455667788,%r11\n\t" //dlsym_proxy
            "cmpq $0xffffffffffffffff,%rdi\n\t" 
            "jz SKIP_DLSYM\n\t"
            "jmpq *%r11\n\t"
            "SKIP_DLSYM:"
            "movq $0x1122334455667788,%r11\n\t" //dlsym
            "jmpq *%r11\n\t"
            );
}


}