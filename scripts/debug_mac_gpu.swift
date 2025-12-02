import Metal
import MetalPerformanceShaders

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
if device.supportsFamily(.metal3) {
  print("Supports Metal 3 features")
} else {
  print("Does not support Metal 3 features")
}
if device.supportsFamily(.common3) {
  print("Supports Common Family 3 features")
} else {
  print("Does not support Common Family 3 features")
}
if device.supportsFamily(.common2) {
  print("Supports Common Family 2 features")
} else {
  print("Does not support Common Family 2 features")
}
if device.supportsFamily(.common1) {
  print("Supports Common Family 1 features")
} else {
  print("Does not support Common Family 1 features")
}
if MPSSupportsMTLDevice(device) {
  print("Supports Metal Performance Shaders")
} else {
  print("Does not support Metal Performance Shaders")
}

