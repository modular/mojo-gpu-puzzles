import Metal
guard let device = MTLCreateSystemDefaultDevice() else {
  print("Error: Could not find Metal device.")
  exit(1)
}
print("Name: \(device.name)")
print("Is Headless: \(device.isHeadless)")
print("Supports Raytracing: \(device.supportsRaytracing)")
print("Supports Dynamic Libraries: \(device.supportsDynamicLibraries)")
print("Max Buffer (GB): \(String(format: "%.2f", Double(device.maxBufferLength) / 1.0e9))")
print("Recommended Max VRAM (GB): \(String(format: "%.2f", Double(device.recommendedMaxWorkingSetSize) / 1.0e9))")