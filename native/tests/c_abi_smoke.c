#include "midas_native.h"

#include <assert.h>
#include <string.h>

int main(void) {
    midas_context* context = 0;
    assert(midas_abi_version() == MIDAS_ABI_VERSION);
    assert(strcmp(midas_version_string(), "0.1.0-foundation") == 0);
    assert(strcmp(midas_status_string(MIDAS_STATUS_OK), "ok") == 0);
    assert(midas_last_error() != 0);
    assert(
        midas_create(
            0, MIDAS_MODEL_V21_SMALL_256, &context) ==
        MIDAS_STATUS_INVALID_ARGUMENT);
    assert(context == 0);
    midas_destroy(0);
    return 0;
}
