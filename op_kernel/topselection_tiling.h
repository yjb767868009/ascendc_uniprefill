#pragma once

#include <cstdint>

constexpr uint32_t TOPSELECTION_MAX_BLOCKS = 2048;
constexpr uint32_t TOPSELECTION_MAX_TOKENS_PER_REQ = 131072;

struct TopSelectionTopPTilingData {
    uint32_t batch;
    uint32_t maxBlockLen;
    float p;
};

struct TopSelectionExpandMaskTilingData {
    uint32_t batch;
    uint32_t maxSeqLen;
    uint32_t blockSize;
    uint32_t attentionSink;
    uint32_t lastQ;
};

