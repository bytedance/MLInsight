/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_ELFPARSER_H
#define MLINSIGHT_ELFPARSER_H

#ifdef __linux

#include <string>
#include <link.h>
#include <vector>
#include <elf.h>
#include <map>
#include "common/Array.h"

#define ELFW(type)    _ElfW (ELF, __ELF_NATIVE_CLASS, type)
namespace mlinsight {
    //todo: Making ELF Parser a more complete class. (Make it possible to parse everything. ELF parser culd be a standalone module)
    class ELFParser {
    public:

        //The path for current ELF file

        explicit ELFParser();

        bool parse(const char *elfPath);

        bool getSecHeader(const int secType, const std::string &secName, Elf64_Shdr &section);

        void getExtSymbolInfo(ssize_t &relaSymId, const char *&funcName, Elf64_Word &bind, Elf64_Word &type, Elf64_Rela *& relaSection);

        Elf64_Addr getRelaOffset(const ssize_t &relaSymId, Elf64_Rela *& relaSection) const;

        ELFParser(ELFParser &) = delete;

        ~ELFParser();

        Elf64_Shdr *secHdr = nullptr;
        ssize_t secHdrSize = 0;

        Elf64_Phdr *progHdr = nullptr;
        ssize_t progHdrSize = 0;

        const char *secHdrStrtbl = nullptr;//Secion Name String Table

        Elf64_Dyn *dynSection = nullptr;
        ssize_t dynSecSize = 0;

        Elf64_Rela *relaPLTSection = nullptr;
        ssize_t relaPLTEntrySize = 0;

        Elf64_Rela *relaDYNSection = nullptr;
        ssize_t relaDYNEntrySize = 0;

        Elf64_Rela *pltSection = nullptr;
        ssize_t pltEntrySize = 0;

        Elf64_Rela *gotSection = nullptr;
        ssize_t gotEntrySize = 0;

        Elf64_Sym *dynSymTbl = nullptr;
        ssize_t dynSymTblSize = 0;

        const char *dynStrTbl = nullptr;
        ssize_t dynStrTblSize = 0;

        Elf64_Ehdr elfHdr;

        bool readSecContent(Elf64_Shdr &urSecHdr, void *&rltAddr, const ssize_t &oriSecSize);


    protected:
        FILE *file = nullptr;

        bool verifyELFImg();

        bool readELFSectionHeader();

        bool readSecStrTable();

        inline bool readRelaPLTEntries();

        inline bool readRelaDYNEntries();

        inline bool readDynStrTable();

        inline bool readDynSymTable();


        bool verifyDynamicLibrary();
    };

}
#endif

#endif
