#!/bin/bash
##===----------------------------------------------------------------------===##
# Apply TheRock rocm-sdk venv fixes (idempotent).
#
# The TheRock 7.14 nightly rocm-sdk wheels have three issues that break the
# in-process HIP/HSA stack used by torch + max.experimental.torch on RDNA4:
#   1. librocprofiler-register.so.0 is internally broken (its own
#      rocprofiler_configure* symbols unresolvable) -> fatal at dlopen,
#      poisoning HIP init. Replaced with a no-op stub.
#   2. torch preloads the sdk's versioned libs while MAX dlopens unversioned
#      names; without libamdhip64.so / libhsa-runtime64.so symlinks the
#      process ends up mixing sdk-7.14 and system-7.2 libs -> HSA init fails.
#   3. librocm_sysdeps_numa dlopens "libnuma.so" (dev symlink, absent on
#      stock Ubuntu) -> GDA/rocSHMEM init noise.
#
# Run after `uv sync` recreates/updates the venv. Requires gcc.
##===----------------------------------------------------------------------===##
set -euo pipefail

VENV="${1:-.venv}"
VENV_SITE="$(ls -d ${VENV}/lib/python3.*/site-packages 2>/dev/null | head -1)"; SDK_CORE="${VENV_SITE}/_rocm_sdk_core/lib"
SDK_LIBS="${VENV_SITE}/_rocm_sdk_libraries/lib"

if [ ! -d "${SDK_CORE}" ]; then
    echo "rocm-sdk not found in ${SDK_CORE}; nothing to fix."
    exit 0
fi

# --- 1. Stub the broken librocprofiler-register -----------------------------
REG="${SDK_CORE}/librocprofiler-register.so.0"
if [ -f "${REG}.orig" ] && [ ! -f "${REG}" ]; then
    echo "restoring original librocprofiler-register (stub removed)"
    mv "${REG}.orig" "${REG}"
fi
if [ -f "${REG}" ] && ! nm -D "${REG}" 2>/dev/null | grep -q " T rocprofiler_configure$"; then
    TMP="$(mktemp -d)"
    cat > "${TMP}/stub.c" <<'EOF'
/* No-op stub for TheRock's broken librocprofiler-register.so.0. */
typedef struct rocprofiler_library_api_table_t { void* api; } rocprofiler_library_api_table_t;
int rocprofiler_configure(void) { return 0; }
int rocprofiler_configure_attach(void) { return 0; }
int rocprofiler_attach(void) { return 0; }
int rocprofiler_detach(void) { return 0; }
int rocprofiler_set_api_table(void) { return 0; }
int rocprofiler_register_attach(void) { return 0; }
int rocprofiler_register_detach(void) { return 0; }
int rocprofiler_register_error_string(void) { return 0; }
int rocprofiler_register_invoke_all_registrations(void) { return 0; }
int rocprofiler_register_invoke_nonpropagated_registrations(void) { return 0; }
int rocprofiler_register_iterate_registration_info(void) { return 0; }
int rocprofiler_register_library_api_table(const rocprofiler_library_api_table_t* t) { (void)t; return 0; }
EOF
    gcc -shared -fPIC -Wl,-soname,librocprofiler-register.so.0 \
        -o "${TMP}/librocprofiler-register.so.0" "${TMP}/stub.c"
    mv "${REG}" "${REG}.orig"
    cp "${TMP}/librocprofiler-register.so.0" "${REG}"
    rm -rf "${TMP}"
    echo "stubbed librocprofiler-register.so.0 (original kept as .orig)"
fi

# --- 2. Unversioned symlinks so torch and MAX load the same libs ------------
ln -sf libamdhip64.so.7 "${SDK_CORE}/libamdhip64.so"
ln -sf libhsa-runtime64.so.1 "${SDK_CORE}/libhsa-runtime64.so"
echo "ensured unversioned libamdhip64.so / libhsa-runtime64.so symlinks"

# --- 3. libnuma.so for librocm_sysdeps_numa ---------------------------------
if [ ! -e "${SDK_CORE}/libnuma.so" ]; then
    NUMA="$(ldconfig -p 2>/dev/null | awk '/libnuma\.so\.1/{print $NF; exit}')"
    if [ -n "${NUMA}" ]; then
        ln -sf "${NUMA}" "${SDK_CORE}/libnuma.so"
        echo "linked libnuma.so -> ${NUMA}"
    fi
fi

# --- 4. sitecustomize: preload librocprofiler-sdk RTLD_GLOBAL ---------------
SC="${VENV_SITE}/sitecustomize.py"
if [ ! -f "${SC}" ]; then
    cat > "${SC}" <<'EOF'
import ctypes, glob, os
_sdk = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rocm_sdk_core", "lib")
for _lib in glob.glob(os.path.join(_sdk, "librocprofiler-sdk.so*")):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
EOF
    echo "wrote sitecustomize.py"
fi

echo "ROCm sdk fixes applied."
