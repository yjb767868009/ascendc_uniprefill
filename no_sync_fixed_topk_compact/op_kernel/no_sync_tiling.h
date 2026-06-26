#pragma once

#include <cstdint>

struct UniPrefillFixedTopKCompactTilingData {
    uint32_t batch;
    uint32_t hiddenSize;
    uint32_t blockSize;
    uint32_t attentionSink;
    uint32_t lastQ;
};


struct UniPrefillFixedTopKTiledSelectTilingData {
    uint32_t batch;
    uint32_t blockSize;
    uint32_t attentionSink;
    uint32_t lastQ;
};

struct UniPrefillFixedTopKTiledCopyTilingData {
    uint32_t batch;
    uint32_t hiddenSize;
    uint32_t blockSize;
    uint32_t hiddenTile;
    uint32_t hiddenTileCount;
};
