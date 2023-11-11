/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_ELFINFO_H
#define MLINSIGHT_ELFINFO_H

#include "RecordingDataStructure.h"

namespace mlinsight {
/**
* ELF image (ELF file in memory) information.
*/
    struct ELFImgInfo {
        uint8_t *pltStartAddr;
        uint8_t *pltSecStartAddr;
        uint8_t *gotStartAddr;
        int64_t firstSymIndex;
        bool valid = false;
    };

    struct ELFSecInfo {
        uint8_t *startAddr;
        Elf64_Word size;
        Elf64_Word entrySize;
    };
}
#endif //MLINSIGHT_ELFIMGINFO_H
