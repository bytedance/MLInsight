/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Original Author <ouadimjamal@gmail.com>
*/

#ifndef MLINSIGHT_PROCINFOPARSER_H
#define MLINSIGHT_PROCINFOPARSER_H

#ifdef __linux__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <linux/limits.h>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <link.h>
#include <elf.h>
#include <set>
#include "trace/type/RecordingDataStructure.h"
#include "common/Array.h"

namespace mlinsight {

    //The following class is declared and defined only under linux.

    /**
     * Represent a line in /prof/{pid}/map
     */
    class PMEntry {
    public:
        //A counter used to mark the number of loaded file ID. This ID is always incremental to support library unloading. -1 means no file has been loaded.

    public:
        uint8_t *addrStart = nullptr;
        uint8_t *addrEnd = nullptr;
        std::string pathNameString;
        FileID globalFileId;
    public:
        enum PERM {
            READ = 3,
            WRITE = 2,
            EXEC = 1,
            PRIVATE = 0
        };

        // PMEntry(uint8_t* addrStart,uint8_t* addrEnd,char* permString, std::string& pathNameString):addrStart(addrStart),
        // addrEnd(addrEnd),pathNameString(pathNameString){
        //     //We define pathNameString to rvalue so that we can save some overhead

        // }


        // end address
        unsigned char permBits = 0; // Is readable

        inline bool isR() const {
            return permBits & (1 << PERM::READ);
        }

        inline bool isW() const {
            return permBits & (1 << PERM::WRITE);
        }

        inline bool isE() const {
            return permBits & (1 << PERM::EXEC);
        }

        inline bool isP() const {
            return permBits & (1 << PERM::PRIVATE);
        }

        inline void setE() {
            permBits |= 1 << PERM::EXEC;
        }

        inline void setW() {
            permBits |= 1 << PERM::WRITE;
        }

        inline void setP() {
            permBits |= 1 << PERM::PRIVATE;
        }

        inline void setR() {
            permBits |= 1 << PERM::READ;
        }

        inline bool operator==(PMEntry &rho) {
            return addrStart == rho.addrStart && addrEnd == rho.addrEnd && permBits == rho.permBits;
        }

        inline void setPermBits(char *permStr) {
            //Parse permission
            permBits = 0;
            if (permStr[0] == 'r') {
                setR();
            }
            if (permStr[1] == 'w') {
                setW();
            }
            if (permStr[2] == 'x') {
                setE();
            }
            if (permStr[3] == 'p') {
                setP();
            }
        }
    };


    /**
     * Group entry by its executable
     */
    class FileEntry {
    public:
        std::string filePath;
        //This counter marks the time when this fileEntry is newly loaded or reloaded.
        ssize_t loadingCounter = -1;
        bool valid = false; //File should be hooked by Scaler or not. It is determined at the first loading time based on fileName.
        bool fileExists = false;
        bool fileUnloaded = false;
        //Used to map fileEntry to pmEntry based on address range.
        //Each element in this vector represents a CONTINOUS driverMemRecord region defined by two pmEntry ids [pair.first,pair,second].
        std::vector<std::pair<ssize_t, ssize_t>> pmEntryRange;
    };

    typedef void (*FileNameCallBack)(const char *pathName, const ssize_t length, const ssize_t fileId);


    /**
     * This class was a helper tool to parse /proc/self/maps
     * Current implementation uses STL API and may not the most efficient way. But it's fine for initialization and the code looks cleaner.
     */
    class PmParser {
    public:
        PmParser();

        explicit PmParser(FILE *saveFolderName, std::string customProcFileName = "");


        /**
         * A convenient way to print /proc/{pid}/maps
         */
        void printPM();

        ssize_t findFileIdByAddr(void *addr);

        /**
         * Return addr is located in which file
         * @param fileId in fileIDMap
         */
        std::vector<PMEntry>::iterator findPmEntryIdByAddr(void *addr);


        ~PmParser();

        /**
         * Parse /proc/{pid}/maps into procMap
         * Multiple invocation will keep the internal pmmap
         */
        bool parsePMMap();

        inline ssize_t getLoadingCounter(){
            return loadingCounter;
        }

        inline void
        getNewFileEntryIds(std::vector<ssize_t> &rltArray, ssize_t lastLoadingCounter, bool mustBeValid = false) {
            for (FileID fileId = 0; fileId < fileEntryArray.size(); ++fileId) {
                FileEntry &curFileEntry = fileEntryArray[fileId];
                bool isValid = curFileEntry.valid || !mustBeValid;
                if (isValid && curFileEntry.loadingCounter > lastLoadingCounter) {
                    //INFO_LOGS("I think %zd has been loaded",callerFileId);
                    rltArray.emplace_back(fileId);
                }
            }
        }

        inline PMEntry &getPmEntry(ssize_t i) {
            PMEntry *rlt = &pmEntryArray[i];
            return *rlt;
        }

        inline FileEntry &getFileEntry(ssize_t i) {
            return fileEntryArray[i];
        }

        inline const FileID getMLInsightFileId() {
            return mlinsightFileID;
        }

        inline const FileID getC10CUDAFileID() {
            return this->libc10_cuda_fileId;
        }

        ssize_t getFileEntryArraySize();

        ssize_t getPythonInterpreterFileId();

        void getAddressRangeByFileId(const FileID &fileId, void *&startAddr, void *&endAddr);

    protected:
        std::vector<char> stringTable;
        std::string customProcFileName;
        std::vector<PMEntry> pmEntryArray;
        std::map<std::string, FileID> fileNameFileIdMap;
        //The index is FileID, the value indicates whether this callerFileId exists in /proc/{pid}/maps.
        //The std::string is fileName and is used to find deleted entries in fileEntryMap
        //The last bool is used to mark whether this callerFileId has been deleted from fileEntryMap
        std::vector<FileEntry> fileEntryArray;
        FILE *fileNameStrTbl{};
        //This counter will increase by 1 when a new file is loaded.
        //The purpose of this counter is to help comparing whether a library is newly installed or not. And also helps to distinguish repeatly load and unload APIs of at the same address.
        ssize_t loadingCounter = -1;
        FileID mlinsightFileID = -1; //The file ID of Scaler itself.
        FileID libc10_cuda_fileId = -1; //The file ID of libc10_cuda.so
        ssize_t pythonIntrepreterFileId = -1; //The file ID of python interpreter
        pthread_mutex_t pmParserLock;

        FILE *openProcFile();

        bool isFileNameValid(int scanfreturnValue, const std::string &pathName, ssize_t curFileId);

        void handleUnloadedFileEntries();

    };

};


#endif
#endif
