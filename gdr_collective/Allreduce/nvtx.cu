// nvtx_wrappers
#include <nvtx3/nvToolsExt.h>

extern "C" {
void nvtxRangePushWrapper(const char *msg) {
    nvtxRangePushA(msg);
}

void nvtxRangePopWrapper() {
    nvtxRangePop();
}
}
