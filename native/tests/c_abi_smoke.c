#include "midas_native.h"

#include <assert.h>
#include <string.h>

int main(void) {
    assert(midas_abi_version() == MIDAS_ABI_VERSION);
    assert(strcmp(midas_version_string(), "0.1.0-foundation") == 0);
    assert(strcmp(midas_status_string(MIDAS_STATUS_OK), "ok") == 0);
    assert(midas_last_error() != 0);
    return 0;
}
