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
        enum PERM {
            READ = 3,
            WRITE = 2,
            EXEC = 1,
            PRIVATE = 0
        };
        // end address
        ssize_t globalFileId = -1;
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

        ssize_t loadingId = -1; //Marks the version of this entry, used to detect entry deletion
        uint8_t *addrStart = nullptr;
        // start address of the segment
        uint8_t *addrEnd = nullptr;
        ssize_t creationLoadingId = -1;//Marks the creation loadingId of this entry. This combined with previous field can be used to detect new file addition.
    };


    /**
     * Group entry by its executable
     */
    class FileEntry {
    public:
        ssize_t pathNameStartIndex = -1;
        ssize_t pathNameEndIndex = -1;
        ssize_t pmEntryNumbers;
        bool valid = false;
        ssize_t loadingId = -1; //Marks the version of this entry, used to detect entry deletion
        ssize_t creationLoadingId = -1;//Marks the creation loadingId of this entry. This combined with previous field can be used to detect new file addition.
        uint8_t *baseStartAddr = nullptr;
        uint8_t *baseEndAddr = nullptr;
        ssize_t recArrFileId=-1;//This corresponds to the loading 

        ssize_t getPathNameLength() {
            return pathNameEndIndex - pathNameStartIndex - 1;
        }
    };

    typedef void (*FileNameCallBack)(const char *pathName, const ssize_t length, const ssize_t fileId);

    /**
     * This class was a helper tool to parse /proc/self/maps
     * Current implementation uses STL API and may not the most efficient way. But it's fine for initialization and the code looks cleaner.
     */
    class PmParser {
    public:
        explicit PmParser(std::string saveFolderName, std::string customProcFileName = "");


        /**
         * A convenient way to print /proc/{pid}/maps
         */
        virtual void printPM();

        virtual ssize_t findFileIdByAddr(void *addr);

        /**
         * Return addr is located in which file.
         * "lo" returns the index such that pmEntryArray[lo].addrStart >= addr
         * @param fileId in fileIDMap
         */
        virtual void findPmEntryIdByAddr(void *addr, ssize_t &lo, bool &found);



        ~PmParser();

        /**
         * Parse /proc/{pid}/maps into procMap
         * Multiple invocation will keep the internal pmmap
         */
        virtual bool parsePMMap();


        virtual inline void
        getNewFileEntryIds(Array<ssize_t> &retArray, bool mustBeValid= false) {
            for (ssize_t i = 0; i < fileEntryArray.getSize(); ++i) {
                if (loadingId == fileEntryArray[i].creationLoadingId && (fileEntryArray[i].valid || !mustBeValid)) {
                    retArray.pushBack(i);
                }
            }
        }

        virtual inline PMEntry &getPmEntry(ssize_t i) {
            PMEntry *ret = &pmEntryArray.get(i);
            return *ret;
        }

        virtual inline FileEntry &getFileEntry(ssize_t i) {
            FileEntry *ret = &fileEntryArray.get(i);
            return *ret;
        }

        const char *getStr(ssize_t strStart);

        ssize_t getFileEntryArraySize();

        inline const ssize_t getMLInsightFileId(){
            return mlinsightFileId;
        }

    protected:
        Array<char> stringTable;
        std::string customProcFileName;
        ssize_t loadingId = -1;
        Array<PMEntry> pmEntryArray;
        Array<FileEntry> fileEntryArray;
        std::string folderName;
        ssize_t mlinsightFileId=-1;

        FILE *openProcFile();

        bool matchWithPreviousFileId(ssize_t curLoadingId, char *pathName,
                                     ssize_t pathNameLen, PMEntry *newPmEntry);

        void createFileEntry(PMEntry *newPmEntry, ssize_t loadingId, char *pathName, ssize_t pathNameLen,
                             ssize_t scanfReadNum);

        void rmDeletedPmEntries(ssize_t loadingId);

        void updateFileBaseAddr();
    };

    extern FileID libc10_cuda_fileId;
    extern void * libc10_cuda_text_begin; // The begin address of c10_cuda_text
    extern void * libc10_cuda_text_end; 
};


#endif
#endif
