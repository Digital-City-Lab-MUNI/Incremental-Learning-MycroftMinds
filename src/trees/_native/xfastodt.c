/* Sparse online oblivious regression tree. No Python or NumPy headers required.
 * Feature subsets are supplied by the binding's NumPy-compatible RNG callback.
 * Allocation counters report requested owned bytes, not allocator overhead/RSS.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef int (*FeatureSelector)(void *, int, int, int *);

typedef struct {
    double edge, count, sum_y, sum_y2;
} Bin;

typedef struct {
    int feature, n_bins;
    Bin *bins;
    double count, sum_x;
    Bin left, right;
} Histogram;

typedef struct Node {
    uint64_t high, low, count, last_attempt;
    int depth, features_ready;
    double sum_y, sum_y2, y_min, y_max, value;
    Histogram *histograms;
    struct Node *next;
} Node;

typedef struct {
    int depth, current_depth, mature, n_features, selected_features;
    int grace_depth, histogram_kind, max_bins, inherit_bins, relative;
    uint64_t grace_period;
    double delta, tau, feature_fraction;
    int *split_features, *selection;
    double *split_values;
    Node **table, *nodes;
    size_t table_capacity, node_count, allocated, peak_allocated;
    FeatureSelector selector;
    void *selector_context;
    char error[192];
} Tree;

static int fail(Tree *tree, const char *message) {
    snprintf(tree->error, sizeof(tree->error), "%s", message);
    return -1;
}

static void *allocate(Tree *tree, size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) {
        fail(tree, "native allocation size overflow");
        return NULL;
    }
    size_t bytes = count * size;
    if (bytes > SIZE_MAX - tree->allocated) {
        fail(tree, "native allocation accounting overflow");
        return NULL;
    }
    void *pointer = calloc(count, size);
    if (!pointer) {
        fail(tree, "native allocation failed");
        return NULL;
    }
    tree->allocated += bytes;
    if (tree->allocated > tree->peak_allocated) tree->peak_allocated = tree->allocated;
    return pointer;
}

static void release(Tree *tree, void *pointer, size_t bytes) {
    if (pointer) {
        free(pointer);
        tree->allocated -= bytes;
    }
}

static uint64_t mix(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static size_t slot_for(uint64_t high, uint64_t low, int depth, size_t capacity) {
    return (size_t)mix(low ^ mix(high) ^ mix((uint64_t)depth)) & (capacity - 1);
}

static int grow_table(Tree *tree) {
    size_t capacity = tree->table_capacity ? tree->table_capacity * 2 : 16;
    if (capacity < tree->table_capacity) return fail(tree, "node table capacity overflow");
    Node **table = allocate(tree, capacity, sizeof(Node *));
    if (!table) return -1;
    for (Node *node = tree->nodes; node; node = node->next) {
        size_t slot = slot_for(node->high, node->low, node->depth, capacity);
        while (table[slot]) slot = (slot + 1) & (capacity - 1);
        table[slot] = node;
    }
    release(tree, tree->table, tree->table_capacity * sizeof(Node *));
    tree->table = table;
    tree->table_capacity = capacity;
    return 0;
}

static Node *lookup(Tree *tree, int depth, uint64_t high, uint64_t low) {
    size_t slot = slot_for(high, low, depth, tree->table_capacity);
    while (tree->table[slot]) {
        Node *node = tree->table[slot];
        if (node->depth == depth && node->high == high && node->low == low) return node;
        slot = (slot + 1) & (tree->table_capacity - 1);
    }
    return NULL;
}

static Node *new_node(Tree *tree, int depth, uint64_t high, uint64_t low, double value) {
    if ((tree->node_count + 1) * 4 >= tree->table_capacity * 3 && grow_table(tree)) return NULL;
    Node *node = allocate(tree, 1, sizeof(Node));
    if (!node) return NULL;
    node->depth = depth;
    node->high = high;
    node->low = low;
    node->value = value;
    node->y_min = INFINITY;
    node->y_max = -INFINITY;
    size_t slot = slot_for(high, low, depth, tree->table_capacity);
    while (tree->table[slot]) slot = (slot + 1) & (tree->table_capacity - 1);
    tree->table[slot] = node;
    node->next = tree->nodes;
    tree->nodes = node;
    tree->node_count++;
    return node;
}

static Node *existing_node(Tree *tree, int depth, uint64_t high, uint64_t low) {
    Node *node = lookup(tree, depth, high, low);
    while (!node && depth > 0) {
        low = (low >> 1) | (high << 63);
        high >>= 1;
        depth--;
        node = lookup(tree, depth, high, low);
    }
    return node;
}

static int allocate_histograms(Tree *tree, Node *node) {
    node->histograms = allocate(tree, (size_t)(tree->selected_features ? tree->selected_features : 1), sizeof(Histogram));
    if (!node->histograms) return -1;
    if (!tree->histogram_kind) {
        for (int i = 0; i < tree->selected_features; i++) {
            node->histograms[i].bins = allocate(tree, (size_t)tree->max_bins + 1, sizeof(Bin));
            if (!node->histograms[i].bins) return -1;
        }
    }
    return 0;
}

static int initialize_features(Tree *tree, Node *node) {
    if (node->features_ready) return 0;
    if (tree->selector(tree->selector_context, tree->n_features,
                       tree->selected_features, tree->selection)) {
        return fail(tree, "feature selection callback failed");
    }
    for (int i = 0; i < tree->selected_features; i++) {
        int feature = tree->selection[i];
        if (feature < 0 || feature >= tree->n_features || (i && feature <= tree->selection[i - 1])) {
            return fail(tree, "feature selector returned invalid indices");
        }
    }
    if (!node->histograms && allocate_histograms(tree, node)) return -1;
    for (int i = 0; i < tree->selected_features; i++) node->histograms[i].feature = tree->selection[i];
    node->features_ready = 1;
    return 0;
}

static void add_target(Bin *bin, double y) {
    bin->count += 1;
    bin->sum_y += y;
    bin->sum_y2 += y * y;
}

static void insert_histogram(Tree *tree, Histogram *histogram, double x, double y) {
    if (tree->histogram_kind) {
        histogram->count += 1;
        histogram->sum_x += x;
        double previous_mean = histogram->count == 1 ? x :
            (histogram->sum_x - x) / (histogram->count - 1);
        add_target(x <= previous_mean ? &histogram->left : &histogram->right, y);
        return;
    }
    int index = 0;
    while (index < histogram->n_bins && histogram->bins[index].edge < x) index++;
    if (histogram->n_bins == 0 || index == histogram->n_bins || (index == 0 && x < histogram->bins[0].edge)) {
        memmove(&histogram->bins[index + 1], &histogram->bins[index],
                (size_t)(histogram->n_bins - index) * sizeof(Bin));
        histogram->bins[index] = (Bin){.edge = x};
        histogram->n_bins++;
    }
    add_target(&histogram->bins[index], y);
    if (histogram->n_bins <= tree->max_bins) return;
    int merge_index = 0;
    double smallest_count = INFINITY;
    for (int i = 0; i + 1 < histogram->n_bins; i++) {
        double count = histogram->bins[i].count + histogram->bins[i + 1].count;
        if (count < smallest_count) {
            smallest_count = count;
            merge_index = i;
        }
    }
    Bin *left = &histogram->bins[merge_index];
    Bin *right = left + 1;
    right->count += left->count;
    right->sum_y = left->sum_y + right->sum_y;
    right->sum_y2 = left->sum_y2 + right->sum_y2;
    memmove(left, right, (size_t)(histogram->n_bins - merge_index - 1) * sizeof(Bin));
    histogram->n_bins--;
}

static int update_stats(Tree *tree, Node *node, const double *features, double target) {
    if (initialize_features(tree, node)) return -1;
    if (node->count == UINT64_MAX) return fail(tree, "node observation count overflow");
    node->count++;
    node->sum_y += target;
    node->sum_y2 += target * target;
    node->value = node->sum_y / (double)node->count;
    if (target < node->y_min) node->y_min = target;
    if (target > node->y_max) node->y_max = target;
    for (int i = 0; i < tree->selected_features; i++) {
        Histogram *histogram = &node->histograms[i];
        insert_histogram(tree, histogram, features[histogram->feature], target);
    }
    return 0;
}

static double variance(double count, double sum, double squared) {
    double mean = sum / count;
    return squared / count - mean * mean;
}

static double split_gain(Tree *tree, Node *node, Bin left, Bin right) {
    if (left.count == 0 || right.count == 0) return 0;
    double parent = variance((double)node->count, node->sum_y, node->sum_y2);
    double children = (left.count / (double)node->count) * variance(left.count, left.sum_y, left.sum_y2)
                    + (right.count / (double)node->count) * variance(right.count, right.sum_y, right.sum_y2);
    double gain = parent - children;
    return tree->relative ? (parent > 0 ? gain / parent : 0) : gain;
}

static void consider_candidate(Tree *tree, Node *node, int feature, double threshold,
                               Bin left, Bin right, double *best, double *second,
                               int *best_feature, double *best_threshold) {
    double gain = split_gain(tree, node, left, right);
    if (gain > *best) {
        *second = *best;
        *best = gain;
        *best_feature = feature;
        *best_threshold = threshold;
    } else if (gain > *second && gain < *best) {
        *second = gain;
    }
}

static int best_split(Tree *tree, Node *node, int *feature, double *threshold) {
    double best = 0, second = 0;
    *feature = -1;
    for (int i = 0; i < tree->selected_features; i++) {
        Histogram *histogram = &node->histograms[i];
        if (tree->histogram_kind) {
            if (histogram->count) consider_candidate(tree, node, histogram->feature,
                histogram->sum_x / histogram->count, histogram->left, histogram->right,
                &best, &second, feature, threshold);
            continue;
        }
        Bin left = {0};
        for (int k = 1; k < histogram->n_bins; k++) {
            Bin previous = histogram->bins[k - 1];
            left.count += previous.count;
            left.sum_y += previous.sum_y;
            left.sum_y2 += previous.sum_y2;
            Bin right = {0};
            for (int j = k; j < histogram->n_bins; j++) {
                right.count += histogram->bins[j].count;
                right.sum_y += histogram->bins[j].sum_y;
                right.sum_y2 += histogram->bins[j].sum_y2;
            }
            double candidate = (previous.edge + histogram->bins[k].edge) / 2;
            consider_candidate(tree, node, histogram->feature, candidate, left, right,
                               &best, &second, feature, threshold);
        }
    }
    double range = tree->relative ? 1 : 0.25 * (node->y_max - node->y_min) * (node->y_max - node->y_min);
    double epsilon = sqrt((range * range) * log(1 / tree->delta) / (2 * (double)node->count));
    return *feature >= 0 && (best - second > epsilon || tree->tau > epsilon);
}

static int inherit_histograms(Tree *tree, Node *parent, Node *child, int feature,
                              double threshold, int left_side) {
    if (!tree->inherit_bins || !parent->histograms) return 0;
    if (allocate_histograms(tree, child)) return -1;
    for (int i = 0; i < tree->selected_features; i++) {
        Histogram *source = &parent->histograms[i];
        Histogram *target = &child->histograms[i];
        if (tree->histogram_kind) {
            if (source->feature != feature) {
                *target = *source;
            } else {
                double mean = source->count ? source->sum_x / source->count : 0;
                /* Match MeanSplitterHistogram.partition_bins, including fresh count/sum_x. */
                if (mean <= threshold) target->left = source->left;
                else target->right = source->right;
            }
        } else {
            for (int j = 0; j < source->n_bins; j++) {
                Bin bin = source->bins[j];
                if (source->feature != feature || ((bin.edge <= threshold) == left_side)) {
                    target->bins[target->n_bins++] = bin;
                }
            }
        }
        target->feature = source->feature;
    }
    return 0;
}

static int materialize_split(Tree *tree, Node *node, int feature, double threshold) {
    uint64_t high = (node->high << 1) | (node->low >> 63);
    uint64_t low = node->low << 1;
    node->last_attempt = node->count;
    Node *left = new_node(tree, node->depth + 1, high, low, node->value);
    Node *right = new_node(tree, node->depth + 1, high, low | 1, node->value);
    if (!left || !right) return -1;
    if (inherit_histograms(tree, node, left, feature, threshold, 1) ||
        inherit_histograms(tree, node, right, feature, threshold, 0)) return -1;
    return 0;
}

static Node *training_node(Tree *tree, int depth, uint64_t high, uint64_t low) {
    Node *node = lookup(tree, depth, high, low);
    if (node) return node;
    if (depth == 0) {
        fail(tree, "missing root node");
        return NULL;
    }
    Node *parent = training_node(tree, depth - 1, high >> 1, (low >> 1) | (high << 63));
    if (!parent) return NULL;
    int feature = tree->split_features[parent->depth];
    if (feature < 0) {
        fail(tree, "missing registered shared split");
        return NULL;
    }
    if (materialize_split(tree, parent, feature, tree->split_values[parent->depth])) return NULL;
    return lookup(tree, depth, high, low);
}

static void route(Tree *tree, const double *features, uint64_t *high, uint64_t *low) {
    /* Independent depth comparisons form a Boolean decision vector first.
     * Pack that vector MSB-first afterward. No pointer traversal or SIMD intrinsics.
     */
    unsigned char decisions[128];
    for (int depth = 0; depth < tree->current_depth; depth++) {
        decisions[depth] = (unsigned char)(features[tree->split_features[depth]] > tree->split_values[depth]);
    }
    *high = 0;
    *low = 0;
    for (int depth = 0; depth < tree->current_depth; depth++) {
        *high = (*high << 1) | (*low >> 63);
        *low = (*low << 1) | decisions[depth];
    }
}

static int update_one(Tree *tree, const double *features, double target) {
    uint64_t high, low;
    route(tree, features, &high, &low);
    Node *node = training_node(tree, tree->current_depth, high, low);
    if (!node || update_stats(tree, node, features, target)) return -1;
    if (node->depth >= tree->depth) return 0;
    int attempted = node->count - node->last_attempt > tree->grace_period ||
        (tree->grace_depth >= 0 && tree->grace_depth < node->depth);
    if (!attempted) return 0;
    int feature;
    double threshold = 0;
    int accepted = best_split(tree, node, &feature, &threshold);
    node->last_attempt = node->count;
    if (!accepted) return 0;
    tree->split_features[node->depth] = feature;
    tree->split_values[node->depth] = threshold;
    if (materialize_split(tree, node, feature, threshold)) return -1;
    tree->current_depth++;
    tree->mature = tree->current_depth >= tree->depth;
    return 0;
}

void xfd_free(Tree *tree) {
    if (!tree) return;
    Node *node = tree->nodes;
    while (node) {
        Node *next = node->next;
        if (node->histograms) {
            for (int i = 0; i < tree->selected_features; i++) free(node->histograms[i].bins);
            free(node->histograms);
        }
        free(node);
        node = next;
    }
    free(tree->table);
    free(tree->split_features);
    free(tree->split_values);
    free(tree->selection);
    free(tree);
}

Tree *xfd_create(int depth, uint64_t grace_period, double delta, double tau,
                 double feature_fraction, int grace_depth, int histogram_kind,
                 int max_bins, int inherit_bins, int relative,
                 FeatureSelector selector, void *selector_context) {
    if (depth < 0 || depth > 128 || !isfinite(delta) || delta <= 0 || delta > 1 ||
        !isfinite(tau) || tau < 0 || !isfinite(feature_fraction) ||
        feature_fraction <= 0 || feature_fraction > 1 || max_bins < 1 ||
        (histogram_kind != 0 && histogram_kind != 1) || !selector ||
        (inherit_bins && feature_fraction != 1)) return NULL;
    Tree *tree = calloc(1, sizeof(Tree));
    if (!tree) return NULL;
    tree->allocated = sizeof(Tree);
    tree->peak_allocated = sizeof(Tree);
    tree->depth = depth;
    tree->mature = depth == 0;
    tree->grace_period = grace_period;
    tree->delta = delta;
    tree->tau = tau;
    tree->feature_fraction = feature_fraction;
    tree->grace_depth = grace_depth;
    tree->histogram_kind = histogram_kind;
    tree->max_bins = max_bins;
    tree->inherit_bins = inherit_bins;
    tree->relative = relative;
    tree->selector = selector;
    tree->selector_context = selector_context;
    tree->split_features = allocate(tree, (size_t)(depth ? depth : 1), sizeof(int));
    tree->split_values = allocate(tree, (size_t)(depth ? depth : 1), sizeof(double));
    if (!tree->split_features || !tree->split_values || grow_table(tree)) {
        xfd_free(tree);
        return NULL;
    }
    for (int i = 0; i < depth; i++) tree->split_features[i] = -1;
    if (!new_node(tree, 0, 0, 0, 0)) {
        xfd_free(tree);
        return NULL;
    }
    return tree;
}

static int validate_features(Tree *tree, const double *features, size_t rows, int columns) {
    tree->error[0] = '\0';
    if (columns < 1 || (tree->n_features && columns != tree->n_features))
        return fail(tree, "feature count differs from the initialized model");
    if (rows > SIZE_MAX / (size_t)columns) return fail(tree, "input dimensions overflow");
    for (size_t i = 0; i < rows * (size_t)columns; i++) {
        if (!isfinite(features[i])) return fail(tree, "features must be finite");
    }
    if (!tree->n_features) {
        int selected = (int)(columns * tree->feature_fraction);
        /* As in Python, a feature fraction can select zero columns. */
        tree->selection = allocate(tree, (size_t)(selected ? selected : 1), sizeof(int));
        if (!tree->selection) return -1;
        tree->n_features = columns;
        tree->selected_features = selected;
    }
    return 0;
}

int xfd_partial_fit(Tree *tree, const double *features, const double *targets, size_t rows, int columns) {
    if (!tree || validate_features(tree, features, rows, columns)) return -1;
    for (size_t i = 0; i < rows; i++) {
        if (!isfinite(targets[i])) return fail(tree, "targets must be finite");
    }
    for (size_t i = 0; i < rows; i++) {
        if (update_one(tree, features + i * (size_t)columns, targets[i])) return -1;
    }
    return 0;
}

int xfd_predict(Tree *tree, const double *features, size_t rows, int columns, double *output) {
    if (!tree || validate_features(tree, features, rows, columns)) return -1;
    for (size_t i = 0; i < rows; i++) {
        uint64_t high, low;
        route(tree, features + i * (size_t)columns, &high, &low);
        Node *node = existing_node(tree, tree->current_depth, high, low);
        if (!node) return -1;
        output[i] = node->value;
    }
    return 0;
}

size_t xfd_memory_bytes(Tree *tree) { return tree ? tree->allocated : 0; }
size_t xfd_peak_memory_bytes(Tree *tree) { return tree ? tree->peak_allocated : 0; }
size_t xfd_node_count(Tree *tree) { return tree ? tree->node_count : 0; }
int xfd_current_depth(Tree *tree) { return tree ? tree->current_depth : 0; }
int xfd_mature(Tree *tree) { return tree ? tree->mature : 0; }
int xfd_spine_depth(Tree *tree) {
    Node *node = lookup(tree, 0, 0, 0);
    int depth = 0;
    while (node && depth < tree->depth) {
        uint64_t high = (node->high << 1) | (node->low >> 63);
        uint64_t low = node->low << 1;
        Node *left = lookup(tree, depth + 1, high, low);
        if (!left) break;
        node = left;
        depth++;
    }
    return depth;
}
int xfd_split_feature(Tree *tree, int depth) { return tree->split_features[depth]; }
double xfd_split_value(Tree *tree, int depth) { return tree->split_values[depth]; }
const char *xfd_error(Tree *tree) { return tree ? tree->error : "native model is closed"; }
int xfd_abi_version(void) { return 2; }
const char *xfd_build_info(void) {
    return "C11; compiler=" __VERSION__ "; ABI=2; reference=FastODT; routing=boolean-vector";
}
