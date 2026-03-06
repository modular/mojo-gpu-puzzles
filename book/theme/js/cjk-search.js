// CJK Search Fallback for mdBook
//
// mdBook's elasticlunr strips non-ASCII characters (including Korean/Chinese/
// Japanese) from both the search index and queries, making CJK search impossible.
// The documentStore still preserves the original text, so this script patches
// elasticlunr.Index.prototype.search to fall back to direct substring search
// against the documentStore when the query contains CJK characters.
//
// This script must load after elasticlunr.min.js (guaranteed by additional-js
// loading after search_js scripts in index.hbs).

(function () {
    'use strict';

    if (typeof elasticlunr === 'undefined') {
        return;
    }

    // CJK character ranges:
    //   U+3000-9FFF : CJK Unified Ideographs, Hiragana, Katakana, CJK symbols
    //   U+AC00-D7AF : Hangul syllables (Korean)
    //   U+F900-FAFF : CJK Compatibility Ideographs
    var CJK_REGEX = /[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]/;

    var originalSearch = elasticlunr.Index.prototype.search;

    elasticlunr.Index.prototype.search = function (query, userConfig) {
        var q = typeof query === 'string' ? query : JSON.stringify(query);
        if (!CJK_REGEX.test(q)) {
            return originalSearch.call(this, query, userConfig);
        }
        return cjkSearch.call(this, query, userConfig);
    };

    function cjkSearch(query, userConfig) {
        var queryStr = typeof query === 'string' ? query : (query.any || '');
        var terms = queryStr.toLowerCase().split(/\s+/).filter(function (t) {
            return t.length > 0;
        });

        if (terms.length === 0) {
            return [];
        }

        // Parse search configuration (bool mode and field boosts)
        var boolMode = 'OR';
        var fieldConfig = {};

        if (userConfig) {
            try {
                var parsed = typeof userConfig === 'string'
                    ? JSON.parse(userConfig)
                    : userConfig;
                if (parsed.bool) boolMode = parsed.bool;
                if (parsed.fields) {
                    for (var f in parsed.fields) {
                        if (parsed.fields.hasOwnProperty(f)) {
                            var b = parsed.fields[f].boost;
                            fieldConfig[f] = { boost: b != null ? b : 1 };
                        }
                    }
                }
            } catch (e) {
                // Malformed config string; fall through to defaults
            }
        }

        // Default field configuration if none provided
        if (Object.keys(fieldConfig).length === 0) {
            fieldConfig = {
                title: { boost: 2 },
                body: { boost: 1 },
                breadcrumbs: { boost: 1 }
            };
        }

        var results = [];
        var docs = this.documentStore.docs;

        for (var ref in docs) {
            if (!docs.hasOwnProperty(ref)) continue;
            var doc = docs[ref];
            if (!doc) continue;

            // Cache lowercased field text per document (avoid repeated toLowerCase)
            var lowerFields = {};
            for (var field in fieldConfig) {
                if (!fieldConfig.hasOwnProperty(field)) continue;
                var text = doc[field];
                lowerFields[field] = text ? text.toLowerCase() : '';
            }

            var totalScore = 0;
            var allTermsFound = true;
            var isAnd = boolMode === 'AND';

            for (var ti = 0; ti < terms.length; ti++) {
                var term = terms[ti];
                var termScore = 0;
                var termFound = false;

                for (var ff in lowerFields) {
                    var lowerText = lowerFields[ff];
                    if (!lowerText) continue;

                    var boost = fieldConfig[ff].boost;

                    // Count substring occurrences
                    var count = 0;
                    var pos = 0;
                    while ((pos = lowerText.indexOf(term, pos)) !== -1) {
                        count++;
                        pos += term.length;
                    }

                    if (count > 0) {
                        termFound = true;
                        var fieldLen = lowerText.length;
                        termScore += (Math.log(1 + count) / Math.sqrt(fieldLen)) * boost * 100;
                    }
                }

                if (!termFound) {
                    allTermsFound = false;
                    if (isAnd) break;
                }
                totalScore += termScore;
            }

            if (isAnd && !allTermsFound) {
                continue;
            }

            if (totalScore > 0) {
                results.push({ ref: ref, score: totalScore, doc: doc });
            }
        }

        results.sort(function (a, b) { return b.score - a.score; });

        return results;
    }
})();
