// ctoc — Count Tokens of Code
// Like cloc, but for Claude tokens.
//
// Uses a greedy longest-match tokenizer built from a reverse-engineered
// vocabulary of 36,495 verified Claude tokens (95-96% accuracy).

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vocab_data.h"

namespace fs = std::filesystem;

// ─── Defaults ────────────────────────────────────────────────────────

static const std::unordered_set<std::string> DEFAULT_EXCLUDED_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "bower_components",
    "__pycache__", ".venv", "venv", "env",
    "build", "dist", "out", "target",
    ".next", ".nuxt",
    "vendor",
    "bazel-bin", "bazel-out", "bazel-testlogs",
    ".cache", ".pytest_cache", ".mypy_cache",
    ".idea", ".vscode", ".vs",
};

static constexpr size_t MAX_FILE_SIZE = 1 * 1024 * 1024; // 1 MB
static constexpr size_t BINARY_CHECK_SIZE = 8192;

// ─── Trie ────────────────────────────────────────────────────────────

struct TrieNode {
    std::unordered_map<unsigned char, TrieNode*> children;
    bool is_terminal = false;

    ~TrieNode() {
        for (auto& [_, child] : children)
            delete child;
    }
};

class Trie {
public:
    Trie() : root_(new TrieNode()) {}
    ~Trie() { delete root_; }

    Trie(const Trie&) = delete;
    Trie& operator=(const Trie&) = delete;

    void insert(const std::string& token) {
        TrieNode* node = root_;
        for (unsigned char c : token) {
            auto it = node->children.find(c);
            if (it == node->children.end()) {
                node->children[c] = new TrieNode();
                node = node->children[c];
            } else {
                node = it->second;
            }
        }
        node->is_terminal = true;
    }

    // Returns the length of the longest match starting at data[pos], or 0.
    size_t longest_match(const std::string& data, size_t pos) const {
        TrieNode* node = root_;
        size_t best = 0;
        for (size_t i = pos; i < data.size(); ++i) {
            auto it = node->children.find(static_cast<unsigned char>(data[i]));
            if (it == node->children.end())
                break;
            node = it->second;
            if (node->is_terminal)
                best = i - pos + 1;
        }
        return best;
    }

private:
    TrieNode* root_;
};

// ─── Tokenizer ───────────────────────────────────────────────────────

static size_t count_tokens(const std::string& text, const Trie& trie) {
    size_t count = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t match_len = trie.longest_match(text, pos);
        if (match_len == 0) {
            // Unknown byte — count as 1 token (single-byte fallback)
            ++pos;
        } else {
            pos += match_len;
        }
        ++count;
    }
    return count;
}

// ─── File discovery ──────────────────────────────────────────────────

static bool is_binary(const std::string& data) {
    size_t check_len = std::min(data.size(), BINARY_CHECK_SIZE);
    for (size_t i = 0; i < check_len; ++i) {
        if (data[i] == '\0')
            return true;
    }
    return false;
}

static std::string read_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct FileEntry {
    fs::path path;
    std::string ext;
    size_t tokens;
};

static std::string get_ext(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

static bool should_exclude_dir(const std::string& dirname,
                               const std::unordered_set<std::string>& excluded) {
    return excluded.count(dirname) > 0;
}

// Check if path starts with a bazel- symlink directory
static bool is_bazel_dir(const fs::path& path) {
    for (const auto& component : path) {
        std::string name = component.string();
        if (name.size() > 6 && name.substr(0, 6) == "bazel-")
            return true;
    }
    return false;
}

static std::vector<FileEntry> discover_files(
    const std::vector<std::string>& paths,
    const std::unordered_set<std::string>& excluded_dirs,
    const std::unordered_set<std::string>& include_exts,
    const Trie& trie)
{
    std::vector<FileEntry> files;

    for (const auto& input_path : paths) {
        fs::path p(input_path);
        std::error_code ec;

        if (!fs::exists(p, ec)) {
            std::cerr << "ctoc: " << input_path << ": No such file or directory\n";
            continue;
        }

        if (fs::is_regular_file(p, ec)) {
            // Single file — always process even if extension unknown
            auto fsize = fs::file_size(p, ec);
            if (ec || fsize > MAX_FILE_SIZE)
                continue;
            std::string ext = get_ext(p);
            if (!include_exts.empty() && include_exts.find(ext) == include_exts.end())
                continue;
            std::string content = read_file(p);
            if (content.empty() || is_binary(content))
                continue;
            if (ext.empty())
                ext = "(none)";
            files.push_back({p, ext, count_tokens(content, trie)});
            continue;
        }

        if (!fs::is_directory(p, ec))
            continue;

        for (auto it = fs::recursive_directory_iterator(
                 p, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); ++it) {

            if (ec) {
                it.increment(ec);
                continue;
            }

            if (it->is_directory()) {
                std::string dirname = it->path().filename().string();
                if (should_exclude_dir(dirname, excluded_dirs) ||
                    is_bazel_dir(it->path().lexically_relative(p))) {
                    it.disable_recursion_pending();
                }
                continue;
            }

            if (!it->is_regular_file())
                continue;

            // Check file size
            auto fsize = it->file_size(ec);
            if (ec || fsize > MAX_FILE_SIZE)
                continue;

            std::string ext = get_ext(it->path());
            if (ext.empty())
                continue;

            if (!include_exts.empty() && include_exts.find(ext) == include_exts.end())
                continue;

            std::string content = read_file(it->path());
            if (content.empty() || is_binary(content))
                continue;

            files.push_back({it->path(), ext, count_tokens(content, trie)});
        }
    }

    return files;
}

// ─── Output formatting ──────────────────────────────────────────────

// Format a number with comma separators: 1234567 -> "1,234,567"
static std::string format_number(size_t n) {
    std::string s = std::to_string(n);
    int insert_pos = static_cast<int>(s.size()) - 3;
    while (insert_pos > 0) {
        s.insert(insert_pos, ",");
        insert_pos -= 3;
    }
    return s;
}

static void print_summary(const std::vector<FileEntry>& files) {
    // Aggregate by extension
    struct ExtStats {
        size_t file_count = 0;
        size_t token_count = 0;
    };
    std::unordered_map<std::string, ExtStats> by_ext;
    size_t total_files = 0;
    size_t total_tokens = 0;

    for (const auto& f : files) {
        by_ext[f.ext].file_count++;
        by_ext[f.ext].token_count += f.tokens;
        total_files++;
        total_tokens += f.tokens;
    }

    // Sort by token count descending
    std::vector<std::pair<std::string, ExtStats>> sorted(by_ext.begin(), by_ext.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.token_count > b.second.token_count; });

    // Calculate column widths
    size_t ext_w = 3; // "Ext"
    for (const auto& [ext, _] : sorted)
        ext_w = std::max(ext_w, ext.size());

    std::string files_str = format_number(total_files);
    std::string tokens_str = format_number(total_tokens);
    size_t files_w = std::max(size_t(5), files_str.size());
    size_t tokens_w = std::max(size_t(6), tokens_str.size());

    for (const auto& [_, stats] : sorted) {
        files_w = std::max(files_w, format_number(stats.file_count).size());
        tokens_w = std::max(tokens_w, format_number(stats.token_count).size());
    }

    size_t total_w = ext_w + 2 + files_w + 2 + tokens_w;
    std::string line(total_w, '-');

    std::cout << line << "\n";
    std::cout << std::left << std::setw(ext_w) << "Ext"
              << "  " << std::right << std::setw(files_w) << "files"
              << "  " << std::setw(tokens_w) << "tokens" << "\n";
    std::cout << line << "\n";

    for (const auto& [ext, stats] : sorted) {
        std::cout << std::left << std::setw(ext_w) << ext
                  << "  " << std::right << std::setw(files_w) << format_number(stats.file_count)
                  << "  " << std::setw(tokens_w) << format_number(stats.token_count) << "\n";
    }

    std::cout << line << "\n";
    std::cout << std::left << std::setw(ext_w) << "SUM"
              << "  " << std::right << std::setw(files_w) << format_number(total_files)
              << "  " << std::setw(tokens_w) << format_number(total_tokens) << "\n";
    std::cout << line << "\n";
}

static void print_by_file(const std::vector<FileEntry>& files) {
    // Sort by tokens descending
    std::vector<const FileEntry*> sorted;
    sorted.reserve(files.size());
    for (const auto& f : files)
        sorted.push_back(&f);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto* a, const auto* b) { return a->tokens > b->tokens; });

    // Calculate column widths
    size_t path_w = 4; // "File"
    size_t ext_w = 3;  // "Ext"
    size_t tokens_w = 6; // "tokens"

    size_t total_tokens = 0;
    for (const auto* f : sorted) {
        path_w = std::max(path_w, f->path.string().size());
        ext_w = std::max(ext_w, f->ext.size());
        tokens_w = std::max(tokens_w, format_number(f->tokens).size());
        total_tokens += f->tokens;
    }

    tokens_w = std::max(tokens_w, format_number(total_tokens).size());

    size_t total_w = path_w + 2 + ext_w + 2 + tokens_w;
    std::string line(total_w, '-');

    std::cout << line << "\n";
    std::cout << std::left << std::setw(path_w) << "File"
              << "  " << std::setw(ext_w) << "Ext"
              << "  " << std::right << std::setw(tokens_w) << "tokens" << "\n";
    std::cout << line << "\n";

    for (const auto* f : sorted) {
        std::cout << std::left << std::setw(path_w) << f->path.string()
                  << "  " << std::setw(ext_w) << f->ext
                  << "  " << std::right << std::setw(tokens_w) << format_number(f->tokens) << "\n";
    }

    std::cout << line << "\n";

    std::string sum_label = "SUM (" + std::to_string(files.size()) + " files)";
    // Pad to fill file + ext columns
    size_t sum_pad = path_w + 2 + ext_w;
    std::cout << std::left << std::setw(sum_pad) << sum_label
              << "  " << std::right << std::setw(tokens_w) << format_number(total_tokens) << "\n";
    std::cout << line << "\n";
}

// ─── Help ────────────────────────────────────────────────────────────

static void print_help() {
    std::cout <<
R"(ctoc - Count Tokens of Code

Like cloc, but counts Claude tokens instead of lines.
Uses a reverse-engineered vocabulary of 36,495 tokens (95-96% accuracy).

USAGE:
    ctoc [OPTIONS] PATH [PATH...]

OPTIONS:
    --by-file            Show per-file token counts
    --exclude-dir DIR    Exclude directory name (repeatable)
    --include-ext EXT    Only include file extension, e.g. .py (repeatable)
    --help               Show this help message

EXAMPLES:
    ctoc src/
    ctoc --by-file main.py utils.py
    ctoc --exclude-dir vendor --exclude-dir test .
    ctoc --include-ext .py --include-ext .js src/
)";
}

// ─── Main ────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    bool by_file = false;
    std::vector<std::string> input_paths;
    std::unordered_set<std::string> extra_excluded_dirs;
    std::unordered_set<std::string> include_exts;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        } else if (arg == "--by-file") {
            by_file = true;
        } else if (arg == "--exclude-dir" && i + 1 < argc) {
            extra_excluded_dirs.insert(argv[++i]);
        } else if (arg == "--include-ext" && i + 1 < argc) {
            std::string ext = argv[++i];
            if (ext.empty()) {
                std::cerr << "ctoc: --include-ext requires a non-empty extension\n";
                return 1;
            }
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext[0] != '.')
                ext = "." + ext;
            include_exts.insert(ext);
        } else if (arg[0] == '-') {
            std::cerr << "ctoc: unknown option: " << arg << "\n";
            std::cerr << "Try 'ctoc --help' for more information.\n";
            return 1;
        } else {
            input_paths.push_back(arg);
        }
    }

    if (input_paths.empty()) {
        std::cerr << "ctoc: no input paths specified\n";
        std::cerr << "Try 'ctoc --help' for more information.\n";
        return 1;
    }

    // Merge excluded dirs
    auto excluded_dirs = DEFAULT_EXCLUDED_DIRS;
    excluded_dirs.insert(extra_excluded_dirs.begin(), extra_excluded_dirs.end());

    // Build trie from embedded vocabulary
    Trie trie;
    for (size_t i = 0; i < VOCAB_COUNT; ++i)
        trie.insert(VOCAB_TOKENS[i]);

    // Discover and tokenize files
    auto files = discover_files(input_paths, excluded_dirs, include_exts, trie);

    if (files.empty()) {
        std::cerr << "ctoc: no files found\n";
        return 1;
    }

    // Output
    if (by_file)
        print_by_file(files);
    else
        print_summary(files);

    return 0;
}
