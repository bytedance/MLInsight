/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/

#ifdef __linux__

#include <sstream>
#include <cassert>
#include <algorithm>
#include <set>
#include <climits>
#include <utility>
#include "common/ProcInfoParser.h"
#include "common/Tool.h"
#include "trace/hook/HookInstaller.h"


#define PROCMAPS_LINE_MAX_LENGTH  (PATH_MAX + 100)
namespace mlinsight {


    PmParser::PmParser(FILE *fileNameStrTbl, std::string customProcFileName) : fileNameStrTbl(fileNameStrTbl),
                                                                               customProcFileName(
                                                                                       customProcFileName) {
        assert(fileNameStrTbl != nullptr);
    }


    bool PmParser::parsePMMap() {
        //printPM();

        FILE * procFile = openProcFile();
        if (!procFile)
            return false;

        //For every parsing, increase pmParsingLoadingCounter by 1
        ++loadingCounter;

        std::string addr1, addr2, perm, offset;

        //Save the filename of this loading
        char procMapLine[4096];
        char permStr[9];
        uint8_t *startAddress;
        uint8_t *endAddress;

        //Clear pmEntryArray
        pmEntryArray.clear();
        //Clear file existence flag
        for (int i = 0; i < fileEntryArray.size(); ++i) {
            fileEntryArray[i].fileExists = false;
            //Since pmEntry has changed, we also need to clear this array and re-parse it again.
            fileEntryArray[i].pmEntryRange.clear();
        }

        while (fgets(procMapLine, sizeof(procMapLine) / sizeof(char), procFile)) {
#ifndef NDEBUG
            //Make sure the buffer is enough
            size_t len = strnlen(procMapLine, sizeof(procMapLine) / sizeof(char));
            if (len != 0 && procMapLine[len] != '\0') {
                fatalErrorS(
                        "Line %s in /proc/{pid}/map exceeded buffer size %lu. Please adjust procMapLine size",
                        procMapLine, sizeof(procMapLine));
            }
#endif
            char pathName[PATH_MAX] = "";
            //Read pmEntry line
            int scanfReadNumber = sscanf(procMapLine, "%p-%p %8s %*s %*s %*s %s", &startAddress, &endAddress, permStr,
                                         pathName);

            //Insert new line into pmEntryArray
            std::string pathNameStr(pathName);
            ssize_t curPmEntryId = pmEntryArray.size();
            PMEntry &curPmEntry = pmEntryArray.emplace_back(PMEntry());
            curPmEntry.addrStart = startAddress;
            curPmEntry.addrEnd = endAddress;
            curPmEntry.setPermBits(permStr);
            curPmEntry.pathNameString = pathNameStr;

            //Find whether current line has been loaded before
            auto fileEntryFindResult = fileNameFileIdMap.find(pathNameStr);

            ssize_t curFileId = -1;
            if (fileEntryFindResult == fileNameFileIdMap.end()) {
                curFileId = fileEntryArray.size();
                //Set fileExist flag to true
                FileEntry &newFileEntry = fileEntryArray.emplace_back(FileEntry());
                newFileEntry.filePath = pathNameStr;
                //newFileEntry.startAddr=startAddress;
                //newFileEntry.endAddr=endAddress;
                newFileEntry.valid = isFileNameValid(scanfReadNumber, pathName, curFileId);
                newFileEntry.loadingCounter = loadingCounter;
                //Insert a new File Entry into fileEntryMap and fileIdFileNameMap
                fileNameFileIdMap[pathNameStr] = curFileId;
                //INFO_LOGS("Found a new file %s",pathNameStr.c_str());
                fprintf(fileNameStrTbl, "%s\n", pathNameStr.c_str());
            } else {
                //This is a pre-existing file or a file that has the same absolute path as before.
                curFileId = fileEntryFindResult->second;
            }
            //Update the file address range
            FileEntry &fileEntry = fileEntryArray[curFileId];
            //FileEntry updated, so we set these flag to true
            fileEntry.fileExists = true; //File still show up in this parsing
            curPmEntry.globalFileId = curFileId; //Map callerFileId to current pmEntry
            if (fileEntry.fileUnloaded) {
                //This file is previously unloaded, so we need to install it again.
                //Set fileEntryloadingCounter to loadingCounter
                fileEntry.loadingCounter = loadingCounter;
            }

            fileEntry.fileUnloaded = false; //File shows up (again) so it is not unloaded

            /*
            * Add this pmEntry to continous address range variable in fileEntry
            */
            if (fileEntry.pmEntryRange.size() == 0) {
                //This struct will be cleared at the beginning of parsePMMap
                fileEntry.pmEntryRange.emplace_back(std::make_pair(curPmEntryId, curPmEntryId));
            } else {
                auto &backRangePair = fileEntry.pmEntryRange.back();
                //Check if the current pmEntry's address range is a continous to backRangePair.second. If no, we will allocate a new pair.
                if (pmEntryArray[backRangePair.second].addrEnd == curPmEntry.addrStart) {
                    //Continous to the previous entry, merge.
                    backRangePair.second = curPmEntryId;
                } else {
                    //Not continous to the previous entry. Push a new pair.
                    fileEntry.pmEntryRange.emplace_back(std::make_pair(curPmEntryId, curPmEntryId));
                }
            }
        }



        //Delete deleted pmEntries
        handleUnloadedFileEntries();


        fclose(procFile);

#ifndef NDEBUG
        //Check whether fileEntry is incremental to ensure there is no overlapping
        //This process involves a lot of string comparison, so it is only performed when NDEBUG is not defined
        //If some assertion is false, then check implementation of this function
        void *curAddr = nullptr;

        for (int i = 0; i < pmEntryArray.size(); ++i) {
            PMEntry &curPmEntry = pmEntryArray[i];
            assert(curAddr <= curPmEntry.addrStart);
            assert(curPmEntry.addrStart < curPmEntry.addrEnd);
            curAddr = curPmEntry.addrEnd;
        }
#endif
        /*
        for(FileID callerFileId=0;callerFileId<fileEntryArray.size();++callerFileId){
            INFO_LOGS("%s",fileEntryArray[callerFileId].filePath.c_str());
            for(ssize_t i=0;i<fileEntryArray[callerFileId].pmEntryRange.size();++i){
                auto& curEntry=fileEntryArray[callerFileId].pmEntryRange[i];
                INFO_LOGS("%p:%p",pmEntryArray[curEntry.first].addrStart,pmEntryArray[curEntry.second].addrEnd);
            }
        }
        */
        return true;
    }


    void PmParser::handleUnloadedFileEntries() {
        for (int i = 0; i < fileEntryArray.size(); ++i) {
            FileEntry &curFileEntry = fileEntryArray[i];
            //Remove all non-updated PLT entries (which means entries no longer exists)
            if (!curFileEntry.fileUnloaded && !curFileEntry.fileExists) {
                //pyModuleNameFileIdMap.erase(curFileEntry.filePath); This erase operation is not necessary because we want to map repeatedly loaded file that has the same absolute path to the same id.
                INFO_LOGS("Scaler finds that file %s has been unloaded", curFileEntry.filePath.c_str());
                curFileEntry.fileUnloaded = true;
            }
        }
    }


    PmParser::~PmParser() {
    }

    void PmParser::printPM() {
        std::stringstream ss;
        ss << "/proc/self/maps";

        std::ifstream ifs(ss.str());
        if (ifs.is_open())
            std::cout << ifs.rdbuf() << std::endl;
    }

    ssize_t PmParser::findFileIdByAddr(void *addr) {
        auto pmEntryIterator = findPmEntryIdByAddr(addr);
        return pmEntryIterator->globalFileId;
    }

    bool comaprePmEntryByStartAddr(const PMEntry &lho, const PMEntry &rho) {
        return lho.addrStart <= rho.addrStart;
    }

    std::vector<PMEntry>::iterator PmParser::findPmEntryIdByAddr(void *addr) {
        PMEntry key;
        key.addrStart = (uint8_t *) addr;
        auto lowerBound = std::lower_bound(pmEntryArray.begin(), pmEntryArray.end(), key, comaprePmEntryByStartAddr);
         --lowerBound;
        assert(lowerBound != pmEntryArray.begin());
        if(!(lowerBound->addrStart <= addr && addr <= lowerBound->addrEnd)){
            this->parsePMMap();
            lowerBound = std::lower_bound(pmEntryArray.begin(), pmEntryArray.end(), key, comaprePmEntryByStartAddr);
            --lowerBound;
        }
        

        assert(lowerBound->addrStart <= addr && addr <= lowerBound->addrEnd);
        return lowerBound;
    }


    FILE *PmParser::openProcFile() {
        FILE * procFile = nullptr;
        if (customProcFileName.empty()) {
            const char *procIdStr = "/proc/self/maps";
            //Using test file rather than real proc filesystem
            procFile = fopen(procIdStr, "rb");
            if (!procFile) {
                ERR_LOGS("Cannot open /proc/self/maps because: %s", strerror(errno));
                return nullptr;
            }
        } else {
            procFile = fopen(customProcFileName.c_str(), "rb");
            if (!procFile) {
                ERR_LOGS("Cannot open %s because: %s", customProcFileName.c_str(), strerror(errno));
                return nullptr;
            }
        }
        return procFile;
    }

    bool PmParser::isFileNameValid(int scanfreturnValue, const std::string &pathName, ssize_t curFileId) {
        //Check the validity of fileEntry
        std::string dirName;
        std::string fileName;
        extractFileName(pathName, dirName, fileName);

        //Check scanf succeeded or not
        if (scanfreturnValue == 3) {
            //DBG_LOG("No file name, do not create file entry");
            return false;
        } else if (pathName[0] == '[') {
            //DBG_LOG("Illegal filename, do not create file entry");
            return false;
        } else if (strStartsWith(fileName, "libmlinsight")) { //|| strStartsWith(fileName, "_scaler")
            DBG_LOGS("libmlinsight name is %s",fileName.c_str());
            //DBG_LOG("Do not create file entry for Scaler library");
            if(strStartsWith(fileName, "libmlinsight.so")){
                //Save current file ID
                assert(mlinsightFileID == -1);
                //Save Scaler's file ID
                mlinsightFileID = curFileId;
            }
            return false;
        } else if (strStartsWith(fileName, "ld-")) {
            //DBG_LOG("Do not hook ld.so library");
            return false;
        } else if (strStartsWith(fileName, "libdl-")) {
            //DBG_LOG("Do not hook ld.so library");
            return false;
        } else if (scanfreturnValue != 4) {
            return false;
        } else if (libc10_cuda_fileId == -1 && fileName.size() == 14 && fileName == "libc10_cuda.so") {
            //DBG_LOG("Do not hook ld.so library");
            //DBG_LOG("Found libc10 cuda fileId");
            libc10_cuda_fileId = curFileId;
        }
        return true;
    }


    ssize_t PmParser::getFileEntryArraySize() {
        return fileEntryArray.size();
    }

    ssize_t PmParser::getPythonInterpreterFileId() {
        /**
         * Python interpreter may have many names. So we try a different approach to find its id.
         */
        if (pythonIntrepreterFileId == -1) {
            void *addr = dlsym(RTLD_DEFAULT, "Py_IsInitialized");
            if (addr != nullptr) {
                pythonIntrepreterFileId = findFileIdByAddr(addr);
//                INFO_LOGS("addr=%p pythonInterpreterFileId=%d",addr, pythonIntrepreterFileId);
//                printPM();
                assert(pythonIntrepreterFileId >= 0);
            }
        }
        return pythonIntrepreterFileId;
    }

    void PmParser::getAddressRangeByFileId(const FileID &fileId, void *&retStartAddr, void *&retEndAddr) {
        FileEntry &curFileEntry = getFileEntry(fileId);
        retStartAddr = (void *) getPmEntry(curFileEntry.pmEntryRange[0].first).addrStart;
        retEndAddr = (void *) getPmEntry(curFileEntry.pmEntryRange.back().second).addrEnd;
    }

    PmParser::PmParser() = default;


}

#endif

