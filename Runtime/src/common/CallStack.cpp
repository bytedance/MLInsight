#include "common/CallStack.h"
#include <unordered_map>
#include "trace/hook/HookInstaller.h"
namespace mlinsight
{
    //todo: Memory in cFrameExtraHeap is never released.
    ObjectPoolHeap<callstack::native::FrameExtra>* cFrameExtraHeap=new ObjectPoolHeap<callstack::native::FrameExtra>(); //A heap to keep all C/C++ extra fields in memory.
    /**
     * Map FrameKey to FrameExtra. This datastructure will only store the latest version of FrameExtra. 
     * Callstacks that uses older versions of FrameExtra must be removed from here right after library unloading. 
     * But callstacks that stores older versions of FrameExtra can still access them because the memory is not freed in cFrameExtraHeap.
    */
    std::unordered_map<CFrameKey_t,CFrameExtra_t*,callstack::Hasher<CFrameKey_t>>* cFrameKeyToFrameExtraMap=new std::unordered_map<CFrameKey_t,CFrameExtra_t*,callstack::Hasher<CFrameKey_t>>();
    std::atomic<ssize_t> globalIdCounter;
    std::unordered_map<CCallStack*,ssize_t,callstack::HasherPtr<CCallStack>,callstack::ComparaterPtr<CCallStack>>* cCallStackRegistery=new std::unordered_map<CCallStack*,ssize_t,callstack::HasherPtr<CCallStack>,callstack::ComparaterPtr<CCallStack>>(); //We do not use virtual method so we cannot fill "Callstack" as key and must define spe
    ObjectPoolHeap<CCallStack>* cCallStackHeap=new ObjectPoolHeap<CCallStack>();
    
    std::atomic<ssize_t> globalCallStackIdCounter=0;

    std::string CCallStack::toString(){
        if(strCache.length()==0){
            std::stringstream ss;
            this->parseAll();
            for (int i = 0; i < this->array.size(); i++)
            {  
                this->array[i].extra->print(ss);
            }
            strCache=ss.str();
        }

        return strCache;
    }

    void CCallStack::print(std::ostream &output)
    {
        output << this->toString(); 
    }

    void CCallStack::parseAll(){
        pthread_mutex_lock(&hookInstallerInstance->dynamicLoadingLock); //Dynamic loading and parsing cannot be overlapped
        // INFO_LOGS("Waiting for debugger pid:%d",getpid());
        // while(!DEBUGGER_CONTINUE){
        //     usleep(1000);
        // }

        auto iter=this->notParsedFrameKeyIndexes.begin();
        for(ssize_t i=0;i<this->notParsedFrameKeyIndexes.size();++i){
            this->parseLine(iter,CFrameKey_t::UNSPECIFIED_LATEST_VERSION);
            assert(this->array[*iter].extra != nullptr);
            ++iter;
        }
        notParsedFrameKeyIndexes.clear();
        assert(this->array[1].extra != nullptr);
        
        pthread_mutex_unlock(&hookInstallerInstance->dynamicLoadingLock); //Dynamic loading and parsing cannot be overlapped
    }
    void CCallStack::parseLine(std::vector<ssize_t>::iterator notParsedFrameKeyIndexIter, ssize_t newVersion)
    {
        pthread_mutex_lock(&hookInstallerInstance->dynamicLoadingLock); //Dynamic loading and parsing cannot be overlapped. This lock is acquired
        ssize_t index=*notParsedFrameKeyIndexIter;
        assert(0<=index && index < this->array.size());

        CFrameKey_t& curFrameKey=this->array[index];
        if (curFrameKey.extra==nullptr)
        {
            // FrameExtra is not parsed yet.
            // Step1: Lookup cache and find if there is a cache value. We do not need to worry about unloaded libraries because they are handled elsewhere.
            assert(curFrameKey.version==CFrameKey_t::UNSPECIFIED_LATEST_VERSION); //Old version symbols should have already been parsed by other parts of the code. In implementation the caller should invoke this first before setting a library.
            auto cacheIter = cFrameKeyToFrameExtraMap->find(curFrameKey);
            if(cacheIter == cFrameKeyToFrameExtraMap->end()){
                //The cache does not exist, allocate a new FrameExtra. Parsing is handled inside FrameExtra.
                CFrameExtra_t* frameExtra = cFrameExtraHeap->alloc();
                new (frameExtra) CFrameExtra_t(curFrameKey.key);
                curFrameKey.extra = frameExtra;

                if(newVersion == CFrameKey_t::UNSPECIFIED_LATEST_VERSION){
                    //This is the CFrameExtra of the latest version of CFrameKey, cache it.
                    cFrameKeyToFrameExtraMap->insert(std::make_pair(curFrameKey,frameExtra));
                }else{
                    //This is parsed for a old version callstack line. So we do not need to cache it inside cFrameKeyToFrameExtraMap.
                }

                //cFrameKeyToFrameExtraMap.insert(std::make_pair(frameKey));
            } else {
                //The cache exists.
                assert(cacheIter->first.version==CFrameKey_t::UNSPECIFIED_LATEST_VERSION); //Element inside cFrameKeyToFrameExtraMap must have version CFrameKey_t::UNSPECIFIED_LATEST_VERSION
                curFrameKey.extra=cacheIter->second;
                if(newVersion == CFrameKey_t::UNSPECIFIED_LATEST_VERSION){
                    //Nothing to do. This FrameKey is the same as cached.
                }else{
                    //This indicates that this cache should be invalidated because library has been unloaded, we remove the element from cache.
                    //Note that this element will still be accessible because it is not freed inside cFrameKeyToFrameExtraMap.
                    cFrameKeyToFrameExtraMap->erase(cacheIter);
                }
            }
            assert(this->array[index].extra != nullptr);
            if(index==1){
                assert(this->array[1].extra != nullptr);
            }

            assert(curFrameKey.extra!=nullptr);//Here curFrameKey should have been parsed already.

            //Update the version number of this key
            curFrameKey.version = newVersion;

            //Remove this index from notParsedFrameKeyIndexes to prevent it from parsed again.
        }
        else
        {
            // Already lazy parsed. Do nothing.
             INFO_LOGS("Line %zd has been parsed, do nothing %p",index, this->array[index].extra);
        }
        pthread_mutex_unlock(&hookInstallerInstance->dynamicLoadingLock);
    }

    void CCallStack::snapshot()
    {   
        void *backtraceArray[CPP_CALL_STACK_LEVEL];
        ssize_t retLevels = backtrace(backtraceArray, CPP_CALL_STACK_LEVEL);
        this->array.clear();
        this->notParsedFrameKeyIndexes.clear();
        strCache.clear();

        for (int i = 0; i < retLevels; ++i)
        {
            this->array.emplace_back(backtraceArray[i], nullptr);
            this->notParsedFrameKeyIndexes.emplace_back(i);
        }

        //Get the calstack id
        // auto insertionIter = cCallStackRegistery->find(*this);
        // if(insertionIter==cCallStackRegistery->end()){
        //     ssize_t newCallStackId=globalIdCounter.fetch_add(1);
        //     this->callstackID=newCallStackId;
        //     cCallStackRegistery->emplace_hint(insertionIter,*this, newCallStackId);
        //     this->isNewCallStackId=true;
        // }else{
        //     this->callstackID=insertionIter->second;
        //     this->isNewCallStackId=false;
        // }
    }

    size_t CCallStack::hash() const{
        size_t hashValue = 0xFFFFFFFF;
            for (int i = 0; i < this->array.size(); ++i) {
                hashValue ^= this->array[i].hash();
            }
        return hashValue;
    }


    CFrameExtra_t::FrameExtra(void* address){

        /**
         * The following code uses mechanism similar to backtrace_symbol
         * https://github.com/bminor/glibc/blob/46b5e98ef6f1b9f4b53851f152ecb8209064b26c/debug/backtracesyms.c#L36
        */

        
        #if __ELF_NATIVE_CLASS == 32
        # define WORD_WIDTH 8
        #else
        /* We assume 64bits.  */
        # define WORD_WIDTH 16
        #endif

        Dl_info info;
        int status;

        /* Fill in the information we can get from `dladdr'.  */
        struct link_map *map;
        status = dladdr1(address, &info,(void**)&map, RTLD_DL_LINKMAP);
        if (status && info.dli_fname && info.dli_fname[0] != '\0')
        {
            /* The load bias is more useful to the user than the load
                address.  The use of these addresses is to calculate an
                address in the ELF file, so its prelinked bias is not
                something we want to subtract out.  */
            info.dli_fbase = (void *)map->l_addr;
        }
        
        if (status && info.dli_fname != NULL && info.dli_fname[0] != '\0')
        {
            if (info.dli_sname == NULL){
                /* We found no symbol name to use, so describe it as
                    relative to the file.  */
                info.dli_saddr = info.dli_fbase;
            }

            if (info.dli_sname == NULL && info.dli_saddr == 0){
                //Only file name, no symbol name, no symbol address
                //this->fileNameStringTableIter= fileNameStringTable.insert(info.dli_fname).first;
                //this->symbolNameStringTableIter= fileNameStringTable.insert(info.dli_sname).first;
                valid=false;
                return;
                // sprintf(retString, "%s(%s) [%p]",
                //                     info.dli_fname ?: "",
                //                     info.dli_sname ?: "",
                //                     address);
            }else
            {
                //Both file name and symbol name presents
                char sign;
                ptrdiff_t offset;
                if (address >= (void *)info.dli_saddr)
                {
                    sign = '+';
                    offset = (uint8_t*)address - (uint8_t*)info.dli_saddr;
                }
                else
                {
                    sign = '-';
                    offset = (uint8_t*)info.dli_saddr - (uint8_t*)address;
                }

                this->fileName=info.dli_fname;
                if(info.dli_sname!=nullptr){
                    this->symbolName= info.dli_sname;
                }
                this->offset=offset;
                this->sign=sign;
                this->valid=true;
                
                // sprintf(retString, "%s(%s%c%#tx) [%p]",
                //             info.dli_fname ?: "",
                //             info.dli_sname ?: "",
                //             // sign, offset, address);
            }
        }
        else{
            //No information at all, only address.
            //sprintf(retString, "[%p]", address);
            this->valid=false;
        }
        
    }


    std::string CFrameExtra_t::toString()
    {   
        if(this->toStringCache.size()==0){
            if(!this->valid){
                toStringCache = "<invalid>";
            }
            char retString[4096];
            snprintf(retString,sizeof(retString)/sizeof(char), "%s(%s%c%#tx)",
                            this->fileName.c_str(),
                            this->symbolName.c_str(),
                            this->sign, this->offset);
            toStringCache =std::string(retString);
        }
        
        return toStringCache;
    }

    void CFrameExtra_t::print(std::ostream& os){
        if(this->valid){
            os<<this->toString()<<std::endl;
        }
    }
}


