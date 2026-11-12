/* Compile/run with make native-check (AddressSanitizer + UBSan + leak checks). */
#include <assert.h>
#include "../src/trees/_native/xfastodt.c"

static int choose_all(void *context, int total, int count, int *output) {
    (void)context;
    assert(count == total);
    for (int i = 0; i < count; i++) output[i] = i;
    return 0;
}

int main(void) {
    for (int iteration = 0; iteration < 40; iteration++) {
        Tree *tree = xfd_create(100, 0, 1, 0.5, 1, -1, iteration % 2, 8,
                                iteration % 3 == 0, 1, choose_all, NULL);
        assert(tree);
        for (int i = 0; i < 1000; i++) {
            double features[3] = {sin(i * 0.17), cos(i * 0.07), i % 13};
            double target = features[0] * 2 + features[1];
            double output;
            assert(xfd_predict(tree, features, 1, 3, &output) == 0);
            assert(isfinite(output));
            assert(xfd_partial_fit(tree, features, &target, 1, 3) == 0);
        }
        assert(xfd_node_count(tree) > 1);
        assert(xfd_peak_memory_bytes(tree) >= xfd_memory_bytes(tree));
        xfd_free(tree);
    }
    puts("Native xFastODT allocation/routing smoke checks passed");
    return 0;
}
