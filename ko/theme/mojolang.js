// This line should be at the top to ensure it runs after highlight.js is loaded
(function () {
    // Register Mojo language
    hljs.registerLanguage("mojo", function () {
        return {
            name: 'Mojo',
            aliases: ['mojo', '🔥'],
            keywords: {
                keyword:
                    'struct fn var let def trait comptime for while if else return continue break ' +
                    'and or not in as is self Self trait ' +
                    'owned in borrowed mut ' +
                    'raises ' +
                    'constrained ' +
                    'from import with try except finally pass ',
                type:
                    'Int UInt Float Bool SIMD Int8 Int16 Int32 Int64 UInt8 UInt16 UInt32 UInt64 ' +
                    'Float16 Float32 Float64 StringLiteral StringRef String True False None ',
                literal: 'True False None'
            },
            contains: [
                {
                    className: 'string',
                    variants: [
                        { begin: /r"/, end: /"/, relevance: 10 },
                        { begin: /"""/, end: /"""/, relevance: 10 },
                        { begin: /"/, end: /"/, illegal: /\n/, contains: [{ begin: /\\./ }] },
                        { begin: /r'/, end: /'/, relevance: 10 },
                        { begin: /'''/, end: /'''/, relevance: 10 },
                        { begin: /'/, end: /'/, illegal: /\n/, contains: [{ begin: /\\./ }] }
                    ]
                },
                {
                    className: 'comment',
                    begin: '#',
                    end: '$',
                    contains: [{ className: 'doctag', begin: '@[A-Za-z]+' }]
                },
                {
                    className: 'comment',
                    begin: '/\\*',
                    end: '\\*/',
                    contains: [{ className: 'doctag', begin: '@[A-Za-z]+' }]
                },
                {
                    className: 'number',
                    variants: [
                        { begin: '\\b0b[01_]+' },
                        { begin: '\\b0o[0-7_]+' },
                        { begin: '\\b0x[0-9a-fA-F_]+' },
                        { begin: '\\b\\d[\\d_]*(\\.([\\d_]*))?([eE][+-]?[\\d_]+)?' }
                    ],
                    relevance: 0
                },
                // Specific keywords that need special handling
                {
                    className: 'keyword',
                    begin: /\b(from|import|alias|fn|def|mut|True|False|None)\b/,
                    relevance: 10
                },
                // Imports - more restricted with beginning-of-line anchors
                {
                    className: 'meta',
                    begin: /^from\s+/,
                    end: /$/,
                    contains: [
                        {
                            className: 'keyword',
                            begin: /^from/,
                            end: /\s+/,
                            excludeEnd: true
                        },
                        {
                            className: 'built_in',
                            begin: /[\w.]+/,
                            end: /\s+import\s+/,
                            excludeEnd: true
                        },
                        {
                            className: 'keyword',
                            begin: /import/,
                            end: /\s+/,
                            excludeEnd: true
                        },
                        {
                            className: 'type',
                            begin: /\b[A-Z][a-zA-Z0-9_]*\b/,
                            relevance: 5
                        },
                        {
                            className: 'punctuation',
                            begin: /,/,
                            end: /\s*/,
                            excludeEnd: true
                        }
                    ],
                    relevance: 10
                },
                {
                    className: 'meta',
                    begin: /^import\s+/,
                    end: /$/,
                    contains: [
                        {
                            className: 'keyword',
                            begin: /^import/,
                            end: /\s+/,
                            excludeEnd: true
                        },
                        {
                            className: 'built_in',
                            begin: /[\w.]+/,
                            end: /$/
                        }
                    ],
                    relevance: 10
                },
                // PascalCase types
                {
                    className: 'type',
                    begin: /\b[A-Z][a-z]+([A-Z][a-z]*)+\b/,
                    relevance: 20
                },
                // Single-word capitalized types
                {
                    className: 'type',
                    begin: /\b[A-Z][a-zA-Z0-9_]*\b/,
                    relevance: 5
                },
                // Generic types with brackets
                {
                    className: 'type',
                    begin: /\b[A-Z][a-zA-Z0-9_]*\s*\[/,
                    end: /\]/,
                    returnBegin: true,
                    contains: [
                        {
                            className: 'type',
                            begin: /\b[A-Z][a-zA-Z0-9_]*\b/
                        },
                        {
                            begin: /\[/,
                            end: /\]/,
                            contains: [
                                {
                                    className: 'type',
                                    begin: /\b[A-Za-z][A-Za-z0-9_]*\b/
                                },
                                {
                                    className: 'keyword',
                                    begin: /:/
                                },
                                {
                                    className: 'keyword',
                                    begin: /=/
                                },
                                {
                                    className: 'keyword',
                                    begin: /mut/
                                },
                                {
                                    className: 'literal',
                                    begin: /\b(True|False)\b/
                                },
                                {
                                    className: 'punctuation',
                                    begin: /,/
                                }
                            ]
                        }
                    ]
                },
                // Function declarations
                {
                    className: 'function',
                    beginKeywords: 'fn def',
                    end: '(:|\\(|$)',
                    excludeEnd: true,
                    contains: [
                        { className: 'title', begin: '[a-zA-Z_]\\w*' },
                        {
                            begin: /\[/, end: /\]/,
                            contains: [
                                { className: 'type', begin: /\b[A-Za-z][A-Za-z0-9_]*\b/ },
                                { className: 'punctuation', begin: /,/ }
                            ]
                        }
                    ]
                },
                // Decorators
                {
                    className: 'meta',
                    begin: /@[a-zA-Z_][a-zA-Z0-9_]*/,
                    relevance: 10
                },
                // Constants
                {
                    className: 'constant',
                    begin: /\b[A-Z][A-Z0-9_]*\b/,
                    relevance: 2
                }
            ]
        };
    });

    // This ensures highlighting runs after the page loads
    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('pre code').forEach(function (block) {
            hljs.highlightElement(block);
        });
    });
})();
