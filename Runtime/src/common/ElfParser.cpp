/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/

#include <elf.h>
#include <link.h>
#include <cstring>
#include <sstream>
#include "common/ELFParser.h"
#include "common/Tool.h"
#include "trace/tool/Math.h"
#include "trace/type/RecordingDataStructure.h"

namespace mlinsight {
    FileID libc10_cuda_fileId;
    void * libc10_cuda_text_begin; 
    void * libc10_cuda_text_end; 
    
    bool ELFParser::parse(const char *elfPath) {
        /**
         * Release previous memory allocations
         */
        if (secHdr) {
            delete secHdr;
            secHdr = nullptr;
        }

        if (file != nullptr) {
            fclose(file);
            file = nullptr;
        }

        file = fopen(elfPath, "rb");
        //INFO_LOGS("Parsing %s", elfPath);
        if (file == NULL) {
            //ERR_LOGS("Can't open ELF file \"%s\", reason: %s", elfPath, strerror(errno));
            return false;
        }

        //Read ELF header
        if (!fread(&elfHdr, 1, sizeof(Elf64_Ehdr), file)) {
            //ERR_LOGS("Failed to read elfHdr because: %s", strerror(errno));
            return false;
        }

        //ELF header contains information the location of section table and program header table
        if (!verifyELFImg()) {
            return false;
        }

        if (!verifyDynamicLibrary()) {
            return false;
        }


        if (!readELFSectionHeader()) {
            return false;
        }

        if (!readSecStrTable()) {
            return false;
        }

        if (!readDynStrTable()) {
            return false;
        }

        if (!readDynSymTable()) {
            return false;
        }

        if (!readRelaPLTEntries()) {
            return false;
        }

        if (!readRelaDYNEntries()) {
            return false;
        }


        return true;
    }

    inline bool ELFParser::verifyELFImg() {
        //Check whether ELF file is valid through magic number
        //This ELF Parser is only used for X86_64
        if (strncmp((const char *) elfHdr.e_ident, ELFMAG, SELFMAG) != 0 ||
            (elfHdr.e_ident[EI_CLASS] != ELFCLASS64) ||
            (elfHdr.e_ident[EI_DATA] != ELFDATA2LSB) ||
            (elfHdr.e_machine != EM_X86_64) ||
            (elfHdr.e_version != 1)) {
            //ERR_LOG("ELF format is not supported, parsing failed");
            return false;
        }
        return true;
    }

    inline bool ELFParser::readELFSectionHeader() {
        //Read all section headers+
        if (fseek(file, elfHdr.e_shoff, SEEK_SET) != 0) {
            //ERR_LOGS("Failed to fseek because: %s", strerror(errno));
            return false;
        }
        secHdrSize = elfHdr.e_shnum;

        //Read all section headers
        if (!secHdr || elfHdr.e_shnum > secHdrSize) {
            secHdr = static_cast<Elf64_Shdr *>(malloc(elfHdr.e_shnum * sizeof(Elf64_Shdr)));
            secHdrSize = elfHdr.e_shnum;
        }
        if (!fread(secHdr, sizeof(ElfW(Shdr)), elfHdr.e_shnum, file)) {
            //ERR_LOGS("Failed to read elfHdr because: %s", strerror(errno));
            return false;
        }
        return true;
    }

    inline bool ELFParser::readSecStrTable() {
        //Read section name string table
        ElfW(Shdr) &strTblSecHdr = secHdr[elfHdr.e_shstrndx];

        //Read string table
        if (fseek(file, strTblSecHdr.sh_offset, SEEK_SET) != 0) {
            //ERR_LOGS("Failed to fseek because: %s", strerror(errno));
            return false;
        }

        secHdrStrtbl = static_cast<const char *>(malloc(sizeof(char) * strTblSecHdr.sh_size));
        if (!secHdrStrtbl) {
            fatalError("Failed to allocate memory for secStrtbl");
            return false;
        }
        if (!fread((void *) secHdrStrtbl, 1, strTblSecHdr.sh_size, file)) {
            //ERR_LOGS("Failed to read secStrtbl because: %s", strerror(errno));
            return false;
        }
        return true;
    }


    ELFParser::~ELFParser() {
        if (secHdr)
            free(secHdr);
        if (secHdrStrtbl)
            free((void *) secHdrStrtbl);
        if (dynSection)
            free(dynSection);
        if (relaPLTSection)
            free(relaPLTSection);
        if (relaDYNSection)
            free(relaDYNSection);
        if (dynSymTbl)
            free(dynSymTbl);
        if (dynStrTbl)
            free((void *) dynStrTbl);
        if (progHdr)
            free(progHdr);
    }

    bool ELFParser::readSecContent(Elf64_Shdr &curSecHdr, void *&retAddr,
                                   const ssize_t &oriSecSize) {

        if (fseek(file, curSecHdr.sh_offset, SEEK_SET) != 0) {
            ERR_LOGS("Failed to fseek because: %s", strerror(errno));
            return false;
        }

        //Do not re-allocate memory if the current memory region is larger

        if (curSecHdr.sh_size > oriSecSize) {
            if (retAddr) {
                free(retAddr);
            }
            retAddr = malloc(curSecHdr.sh_size);
            if (!retAddr) {
                fatalError("Cannot allocate memory for section header");
            }
        } else {
            //Use the old memory section
        }


        if (!fread(retAddr, curSecHdr.sh_size, 1, file)) {
            ERR_LOGS("Failed to read section header because: %s", strerror(errno));
            return false;
        }

        return true;

    }

    ELFParser::ELFParser() {

    }

    void ELFParser::getExtSymbolInfo(ssize_t &relaSymId, const char *&funcName, Elf64_Word &bind, Elf64_Word &type,Elf64_Rela *& relaSection) {
        ssize_t relIdx = ELF64_R_SYM(relaSection[relaSymId].r_info);
        ssize_t strIdx = dynSymTbl[relIdx].st_name;
        //DBG_LOGS("%s:%zd", dynStrTbl + strIdx,strIdx);
        funcName = dynStrTbl + strIdx;
        bind = ELF64_ST_BIND(dynSymTbl[relIdx].st_info);
        type = ELF64_ST_TYPE(dynSymTbl[relIdx].st_info);
    }

    


    inline bool ELFParser::readDynStrTable() {
        Elf64_Shdr curHeader;
        if (!getSecHeader(SHT_STRTAB, ".dynstr", curHeader)) {
            ERR_LOG("Cannot read .dynstr header");
            return false;
        }

        if (!readSecContent(curHeader, (void *&) dynStrTbl, dynStrTblSize)) {
            ERR_LOG("Cannot read .dynstr");
            return false;
        }
        dynStrTblSize = curHeader.sh_size;
        return true;
    }

    bool ELFParser::readDynSymTable() {
        Elf64_Shdr curHeader;
        if (!getSecHeader(SHT_DYNSYM, ".dynsym", curHeader)) {
            ERR_LOG("Cannot read .dynstr header");
            return false;
        }

        if (!readSecContent(curHeader, (void *&) dynSymTbl,
                            dynSymTblSize * sizeof(Elf64_Sym))) {
            ERR_LOG("Cannot read .dynsym");
            return false;
        }
        //dynSymTbl = static_cast<Elf64_Sym *>
        assert(curHeader.sh_entsize == sizeof(Elf64_Sym));
        dynSymTblSize = curHeader.sh_size / curHeader.sh_entsize;
        return true;
    }

    inline bool ELFParser::readRelaPLTEntries() {
        Elf64_Shdr curHeader;
        if (!getSecHeader(SHT_RELA, ".rela.plt", curHeader)) {
            //ERR_LOG("Cannot read .rela.plt header");
            return false;
        }

        if (!readSecContent(curHeader, (void *&) relaPLTSection, relaPLTEntrySize)) {
            ERR_LOG("Cannot read .rela.plt");
            return false;
        }
        assert(curHeader.sh_entsize == sizeof(Elf64_Rela));
        relaPLTEntrySize = curHeader.sh_size / curHeader.sh_entsize;
        return true;
    }

    inline bool ELFParser::readRelaDYNEntries() {
        Elf64_Shdr curHeader;
        if (!getSecHeader(SHT_RELA, ".rela.dyn", curHeader)) {
            ERR_LOG("Cannot read .rela.dyn header");
            return false;
        }

        if (!readSecContent(curHeader, (void *&) relaDYNSection, relaDYNEntrySize)) {
            ERR_LOG("Cannot read .rela.plt");
            return false;
        }
        assert(curHeader.sh_entsize == sizeof(Elf64_Rela));
        relaDYNEntrySize = curHeader.sh_size / curHeader.sh_entsize;
        return true;
    }


    bool ELFParser::getSecHeader(const int secType, const std::string &secName, Elf64_Shdr &section) {
        for (int i = 0; i < secHdrSize; ++i) {
            if (secHdr[i].sh_type == secType &&
                strncmp(secHdrStrtbl + secHdr[i].sh_name, secName.c_str(), secName.size()) == 0) {
                section = secHdr[i];
                return true;
            }
        }
        return false;
    }

    Elf64_Addr ELFParser::getRelaOffset(const ssize_t &relaSymId, Elf64_Rela *& relaSection) const {
        return relaSection[relaSymId].r_offset;
    }

//    void *ELFParser::parseSecLoc(Elf64_Shdr &curHeader, uint8_t *baseStartAddr, uint8_t *possibleStartAddr,
//                                 uint8_t *possibleEndAddr) {
//        assert(possibleStartAddr != nullptr);
//        assert(possibleEndAddr != nullptr);

//        void *ret = autoAddBaseAddr((uint8_t*)curHeader.sh_addr,baseStartAddr,possibleStartAddr,possibleEndAddr);


#ifndef NDEBUG
        //        //Compare bit by bit between ret and content read from file.
        //
        //        uint8_t *pltFromFile = nullptr;
        //        if (!readSecContent(curHeader,
        //                            reinterpret_cast<void *&>(pltFromFile), 0)) {
        //            fatalError("Cannot parse .plt from elf file");
        //        }
        //
        //        for (int i = 0; i < min<Elf64_Xword>(16 * 4, curHeader.sh_size); ++i) {
        //            assert(pltFromFile[i] == *((uint8_t *) ret + i));
        //        }
        //        free(pltFromFile);
#endif
//        return ret;
//    }



    bool ELFParser::verifyDynamicLibrary() {

        if (!progHdr || progHdrSize < elfHdr.e_phnum) {
            //Read all section headers
            progHdr = static_cast<Elf64_Phdr *>(malloc(elfHdr.e_phnum * sizeof(Elf64_Phdr)));
            progHdrSize = elfHdr.e_phnum;
        }

        if (!fread(progHdr, sizeof(Elf64_Phdr), progHdrSize, file)) {
            ERR_LOGS("Failed to read elfHdr because: %s", strerror(errno));
            return false;
        }
        bool foundProgHdr = false;
        for (int i = 0; i < progHdrSize; ++i) {
            if (progHdr[i].p_type == PT_DYNAMIC) {
                foundProgHdr = true;
                return true;
            }
        }

        return foundProgHdr;
    }


}
