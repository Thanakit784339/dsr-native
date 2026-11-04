// Vulkan backend (Android). Stage 2a: real GPU images for the game's render targets, clears and
// texture-to-texture copies, and presentation of the window texture to a swapchain on the device's
// screen. Draws are not executed yet (they need Maxwell shader translation).
//
// Execution is deliberately simple for now: each submitted command list is recorded into one Vulkan
// command buffer, submitted, and waited on. That keeps NVN syncs trivially correct (they are always
// already signaled) at the cost of CPU/GPU overlap, which does not matter until draws exist.
#if defined(__ANDROID__)

#define VK_USE_PLATFORM_ANDROID_KHR
#include <android/native_window.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "common/log.h"
#include "hle/nvn/backend.h"
#include "hle/video.h"

namespace dsr::hle::nvn {
namespace {
constexpr const char* kTag = "vulkan";

bool IsDepthFormat(int f) { return f >= 49 && f <= 54; }
bool HasDepth(int f) { return f >= 50 && f <= 54; }
bool HasStencil(int f) { return f == 49 || f == 53 || f == 54; }
VkImageAspectFlags AspectOf(int f) {
    if (!IsDepthFormat(f)) return VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageAspectFlags a = 0;
    if (HasDepth(f)) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (HasStencil(f)) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a;
}

// NVN format -> candidate Vulkan formats, best first. Only formats usable as render targets or copy
// endpoints matter at this stage; sampled-only formats (BCn, ASTC) arrive with texture upload.
std::vector<VkFormat> Candidates(int f) {
    switch (f) {
        case 1: return {VK_FORMAT_R8_UNORM};
        case 2: return {VK_FORMAT_R8_SNORM};
        case 3: return {VK_FORMAT_R8_UINT};
        case 4: return {VK_FORMAT_R8_SINT};
        case 5: return {VK_FORMAT_R16_SFLOAT};
        case 6: return {VK_FORMAT_R16_UNORM};
        case 7: return {VK_FORMAT_R16_SNORM};
        case 8: return {VK_FORMAT_R16_UINT};
        case 9: return {VK_FORMAT_R16_SINT};
        case 10: return {VK_FORMAT_R32_SFLOAT};
        case 11: return {VK_FORMAT_R32_UINT};
        case 12: return {VK_FORMAT_R32_SINT};
        case 13: return {VK_FORMAT_R8G8_UNORM};
        case 14: return {VK_FORMAT_R8G8_SNORM};
        case 15: return {VK_FORMAT_R8G8_UINT};
        case 16: return {VK_FORMAT_R8G8_SINT};
        case 17: return {VK_FORMAT_R16G16_SFLOAT};
        case 18: return {VK_FORMAT_R16G16_UNORM};
        case 19: return {VK_FORMAT_R16G16_SNORM};
        case 20: return {VK_FORMAT_R16G16_UINT};
        case 21: return {VK_FORMAT_R16G16_SINT};
        case 22: return {VK_FORMAT_R32G32_SFLOAT};
        case 23: return {VK_FORMAT_R32G32_UINT};
        case 24: return {VK_FORMAT_R32G32_SINT};
        case 37: return {VK_FORMAT_R8G8B8A8_UNORM};
        case 38: return {VK_FORMAT_R8G8B8A8_SNORM};
        case 39: return {VK_FORMAT_R8G8B8A8_UINT};
        case 40: return {VK_FORMAT_R8G8B8A8_SINT};
        case 41: return {VK_FORMAT_R16G16B16A16_SFLOAT};
        case 42: return {VK_FORMAT_R16G16B16A16_UNORM};
        case 43: return {VK_FORMAT_R16G16B16A16_SNORM};
        case 44: return {VK_FORMAT_R16G16B16A16_UINT};
        case 45: return {VK_FORMAT_R16G16B16A16_SINT};
        case 46: return {VK_FORMAT_R32G32B32A32_SFLOAT};
        case 47: return {VK_FORMAT_R32G32B32A32_UINT};
        case 48: return {VK_FORMAT_R32G32B32A32_SINT};
        case 49: return {VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
        case 50: return {VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT};
        case 51: return {VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT};
        case 52: return {VK_FORMAT_D32_SFLOAT};
        case 53: return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
        case 54: return {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
        case 55:
        case 56: return {VK_FORMAT_R8G8B8A8_SRGB};
        case 57: return {VK_FORMAT_R4G4B4A4_UNORM_PACK16, VK_FORMAT_R8G8B8A8_UNORM};
        case 58: return {VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_R8G8B8A8_UNORM};
        case 59: return {VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_R8G8B8A8_UNORM};
        case 60: return {VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_R8G8B8A8_UNORM};
        case 61: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32};
        case 62: return {VK_FORMAT_A2B10G10R10_UINT_PACK32};
        case 63: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_R16G16B16A16_SFLOAT};
        case 64: return {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, VK_FORMAT_R16G16B16A16_SFLOAT};
        default: return {};
    }
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    int nvn_format = 0;
    int width = 0, height = 0;
    VkImageAspectFlags aspect = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkFormatFeatureFlags features = 0;
    bool written = false;  // has the GPU put anything in it yet
};

class VulkanBackend final : public Backend {
public:
    const char* Name() const override { return "vulkan"; }

    void Execute(const CmdList& cmds) override {
        std::lock_guard<std::mutex> lk(m_);
        if (!EnsureDevice()) return;
        bool begun = false;
        auto begin = [&] {
            if (begun) return;
            begun = true;
            BeginCommands();
        };
        std::vector<TexRef> rts;
        int num_rt = 0;
        for (const Cmd& c : cmds) {
            switch (c.type) {
                case CmdType::SetRenderTargets:
                    num_rt = c.num_rt;
                    rts = c.rts ? *c.rts : std::vector<TexRef>{};
                    LogTargets(rts, num_rt);
                    break;
                case CmdType::ClearColor:
                    if (c.mask == 0 || c.index < 0 || c.index >= num_rt) break;
                    if (Image* img = GetImage(rts[c.index])) {
                        begin();
                        ClearColorImage(*img, c.color, c.kind);
                    }
                    break;
                case CmdType::ClearTexture:
                    if (c.mask == 0) break;
                    if (Image* img = GetImage(c.dst)) {
                        begin();
                        ClearColorImage(*img, c.color, c.kind);
                    }
                    break;
                case CmdType::ClearDepthStencil: {
                    if (rts.size() <= size_t(num_rt)) break;
                    Image* img = GetImage(rts[num_rt]);
                    if (!img) break;
                    VkImageAspectFlags a = 0;
                    if (c.depth_mask && HasDepth(img->nvn_format)) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
                    if (c.stencil_mask && HasStencil(img->nvn_format)) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
                    a &= img->aspect;
                    if (!a) break;
                    begin();
                    Transition(*img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                    VkClearDepthStencilValue v{std::clamp(c.depth, 0.0f, 1.0f), uint32_t(c.stencil)};
                    VkImageSubresourceRange range{a, 0, 1, 0, 1};
                    vkCmdClearDepthStencilImage(cmd_, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
                    img->written = true;
                    break;
                }
                case CmdType::CopyTexture: {
                    Image* src = GetImage(c.src);
                    Image* dst = GetImage(c.dst);
                    if (!src || !dst || src == dst) break;
                    begin();
                    CopyImage(*src, c.src_region, *dst, c.dst_region, c.flags);
                    break;
                }
                case CmdType::Draw: break;
            }
        }
        if (begun) SubmitAndWait(VK_NULL_HANDLE, VK_NULL_HANDLE);
    }

    void Present(const TexRef& tex) override {
        std::lock_guard<std::mutex> lk(m_);
        if (!EnsureDevice() || !EnsureSwapchain()) return;
        uint32_t idx = 0;
        VkResult r = vkAcquireNextImageKHR(device_, swapchain_, 1000000000ull, acquire_sem_, VK_NULL_HANDLE, &idx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) {
            swapchain_dirty_ = true;
            return;
        }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            LogOnce("acquire", "vkAcquireNextImageKHR failed: %d", r);
            return;
        }
        BeginCommands();
        VkImage sc = sc_images_[idx];
        Barrier(sc, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue black{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd_, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);

        Image* src = tex.valid ? FindImage(tex) : nullptr;
        if (src && src->written && !IsDepthFormat(src->nvn_format)) {
            // Fit the game's frame into the screen, keeping its aspect ratio.
            int sw = src->width, sh = src->height;
            int dw = int(sc_extent_.width), dh = int(sc_extent_.height);
            int w = dw, h = int(int64_t(dw) * sh / std::max(1, sw));
            if (h > dh) {
                h = dh;
                w = int(int64_t(dh) * sw / std::max(1, sh));
            }
            int x = (dw - w) / 2, y = (dh - h) / 2;
            Transition(*src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {sw, sh, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[0] = {x, y, 0};
            blit.dstOffsets[1] = {x + w, y + h, 1};
            vkCmdBlitImage(cmd_, src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            if (!logged_first_image_) {
                logged_first_image_ = true;
                LOGI(kTag, "showing the game's frame: %dx%d (format %d) -> %dx%d at %d,%d", sw, sh, src->nvn_format, w,
                     h, x, y);
            }
        }
        Barrier(sc, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        SubmitAndWait(acquire_sem_, render_sem_);

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &render_sem_;
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain_;
        pi.pImageIndices = &idx;
        r = vkQueuePresentKHR(queue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || r == VK_ERROR_SURFACE_LOST_KHR)
            swapchain_dirty_ = true;
        else if (r != VK_SUCCESS)
            LogOnce("present", "vkQueuePresentKHR failed: %d", r);
        if (++presented_ == 1) LOGI(kTag, "first frame on screen");
    }

private:
    // ---- device ---------------------------------------------------------------------------------------------
    bool EnsureDevice() {
        if (device_) return true;
        if (init_failed_) return false;
        init_failed_ = true;  // cleared on success
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "DSR Native";
        app.apiVersion = VK_API_VERSION_1_1;
        const char* inst_ext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = 2;
        ici.ppEnabledExtensionNames = inst_ext;
        VkResult r = vkCreateInstance(&ici, nullptr, &instance_);
        if (r != VK_SUCCESS) {
            LOGE(kTag, "vkCreateInstance failed: %d — continuing without graphics", r);
            return false;
        }
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance_, &n, nullptr);
        if (!n) {
            LOGE(kTag, "no Vulkan GPU found — continuing without graphics");
            return false;
        }
        std::vector<VkPhysicalDevice> pds(n);
        vkEnumeratePhysicalDevices(instance_, &n, pds.data());
        phys_ = pds[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys_, &props);
        LOGI(kTag, "GPU: %s, Vulkan %u.%u.%u, driver 0x%x", props.deviceName, VK_VERSION_MAJOR(props.apiVersion),
             VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion), props.driverVersion);
        vkGetPhysicalDeviceMemoryProperties(phys_, &mem_props_);

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qf.data());
        family_ = UINT32_MAX;
        for (uint32_t i = 0; i < qn; i++)
            if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                family_ = i;
                break;
            }
        if (family_ == UINT32_MAX) {
            LOGE(kTag, "no graphics queue — continuing without graphics");
            return false;
        }
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        const char* dev_ext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dev_ext;
        r = vkCreateDevice(phys_, &dci, nullptr, &device_);
        if (r != VK_SUCCESS) {
            LOGE(kTag, "vkCreateDevice failed: %d — continuing without graphics", r);
            device_ = VK_NULL_HANDLE;
            return false;
        }
        vkGetDeviceQueue(device_, family_, 0, &queue_);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family_;
        vkCreateCommandPool(device_, &pci, nullptr, &pool_);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &cai, &cmd_);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(device_, &fci, nullptr, &fence_);
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device_, &sci, nullptr, &acquire_sem_);
        vkCreateSemaphore(device_, &sci, nullptr, &render_sem_);
        init_failed_ = false;
        LOGI(kTag, "device ready (queue family %u)", family_);
        return true;
    }

    uint32_t FindMemory(uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mem_props_.memoryTypes[i].propertyFlags & want) == want) return i;
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; i++)
            if (bits & (1u << i)) return i;
        return 0;
    }

    // ---- swapchain ----------------------------------------------------------------------------------------
    void DestroySwapchain(bool and_surface) {
        if (device_) vkDeviceWaitIdle(device_);
        if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
        sc_images_.clear();
        if (and_surface && surface_) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
    }

    bool EnsureSwapchain() {
        void* w = dsr::video::GetNativeWindow();
        if (w != window_) {
            if (window_) LOGI(kTag, "window changed (%p -> %p)", window_, w);
            DestroySwapchain(true);
            window_ = w;
        }
        if (!window_) return false;
        if (!surface_) {
            VkAndroidSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
            sci.window = static_cast<ANativeWindow*>(window_);
            VkResult r = vkCreateAndroidSurfaceKHR(instance_, &sci, nullptr, &surface_);
            if (r != VK_SUCCESS) {
                LogOnce("surface", "vkCreateAndroidSurfaceKHR failed: %d", r);
                surface_ = VK_NULL_HANDLE;
                return false;
            }
        }
        if (swapchain_ && !swapchain_dirty_) return true;

        VkSurfaceCapabilitiesKHR caps;
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surface_, &caps) != VK_SUCCESS) return false;
        uint32_t nf = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &nf, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(nf);
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &nf, formats.data());
        if (formats.empty()) return false;
        VkSurfaceFormatKHR fmt = formats[0];
        for (const auto& f : formats)
            if (f.format == VK_FORMAT_R8G8B8A8_UNORM) {
                fmt = f;
                break;
            }

        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX) {
            auto* aw = static_cast<ANativeWindow*>(window_);
            extent = {uint32_t(ANativeWindow_getWidth(aw)), uint32_t(ANativeWindow_getHeight(aw))};
        }
        // We present with an identity transform and let the compositor rotate if the panel is mounted
        // sideways; in that case the surface extent is reported in the panel's native orientation.
        if (caps.currentTransform & (VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR | VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR))
            std::swap(extent.width, extent.height);
        VkSurfaceTransformFlagBitsKHR transform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                                                      ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                                                      : caps.currentTransform;
        VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        for (VkCompositeAlphaFlagBitsKHR a : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
            if (caps.supportedCompositeAlpha & a) {
                alpha = a;
                break;
            }
        uint32_t count = std::max(caps.minImageCount, 3u);
        if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);

        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface_;
        ci.minImageCount = count;
        ci.imageFormat = fmt.format;
        ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = extent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = transform;
        ci.compositeAlpha = alpha;
        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE;
        ci.oldSwapchain = swapchain_;
        VkSwapchainKHR sc = VK_NULL_HANDLE;
        VkResult r = vkCreateSwapchainKHR(device_, &ci, nullptr, &sc);
        DestroySwapchain(false);
        if (r != VK_SUCCESS) {
            LogOnce("swapchain", "vkCreateSwapchainKHR failed: %d", r);
            return false;
        }
        swapchain_ = sc;
        swapchain_dirty_ = false;
        sc_extent_ = extent;
        uint32_t ni = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &ni, nullptr);
        sc_images_.resize(ni);
        vkGetSwapchainImagesKHR(device_, swapchain_, &ni, sc_images_.data());
        LOGI(kTag, "swapchain %ux%u, format %d, %u images (surface transform 0x%x)", extent.width, extent.height,
             fmt.format, ni, caps.currentTransform);
        return true;
    }

    // ---- images ---------------------------------------------------------------------------------------------
    Image* FindImage(const TexRef& t) {
        auto it = images_.find(t.addr);
        if (it == images_.end()) return nullptr;
        Image& img = it->second;
        if (img.nvn_format != t.tex.format || img.width != t.tex.width || img.height != t.tex.height) return nullptr;
        return &img;
    }

    VkFormat PickFormat(int nvn_format, VkFormatFeatureFlags* features) {
        auto cached = format_cache_.find(nvn_format);
        if (cached != format_cache_.end()) {
            *features = cached->second.second;
            return cached->second.first;
        }
        VkFormatFeatureFlags need = IsDepthFormat(nvn_format) ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
                                                              : VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        VkFormat chosen = VK_FORMAT_UNDEFINED;
        VkFormatFeatureFlags feats = 0;
        for (VkFormat f : Candidates(nvn_format)) {
            VkFormatProperties p;
            vkGetPhysicalDeviceFormatProperties(phys_, f, &p);
            if ((p.optimalTilingFeatures & need) == need) {
                chosen = f;
                feats = p.optimalTilingFeatures;
                break;
            }
        }
        if (chosen == VK_FORMAT_UNDEFINED) LOGW(kTag, "NVN format %d has no usable Vulkan format yet", nvn_format);
        format_cache_[nvn_format] = {chosen, feats};
        *features = feats;
        return chosen;
    }

    Image* GetImage(const TexRef& t) {
        if (!t.valid || t.tex.width <= 0 || t.tex.height <= 0) return nullptr;
        if (Image* existing = FindImage(t)) return existing;
        VkFormatFeatureFlags feats = 0;
        VkFormat vf = PickFormat(t.tex.format, &feats);
        if (vf == VK_FORMAT_UNDEFINED) return nullptr;

        auto old = images_.find(t.addr);
        if (old != images_.end()) {
            vkQueueWaitIdle(queue_);
            DestroyImage(old->second);
            images_.erase(old);
        }

        Image img;
        img.nvn_format = t.tex.format;
        img.vk_format = vf;
        img.width = t.tex.width;
        img.height = t.tex.height;
        img.aspect = AspectOf(t.tex.format);
        img.features = feats;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = vf;
        ci.extent = {uint32_t(img.width), uint32_t(img.height), 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (IsDepthFormat(t.tex.format))
            ci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        else if (feats & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
            ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device_, &ci, nullptr, &img.image) != VK_SUCCESS) {
            LogOnce("image", "vkCreateImage failed for %dx%d format %d", img.width, img.height, img.nvn_format);
            return nullptr;
        }
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device_, img.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device_, &ai, nullptr, &img.memory) != VK_SUCCESS) {
            vkDestroyImage(device_, img.image, nullptr);
            LogOnce("memory", "out of GPU memory for %dx%d format %d", img.width, img.height, img.nvn_format);
            return nullptr;
        }
        vkBindImageMemory(device_, img.image, img.memory, 0);
        if (images_created_++ < 48)
            LOGI(kTag, "image %dx%d NVN format %d -> Vulkan format %d (storage 0x%llx)", img.width, img.height,
                 img.nvn_format, vf, (unsigned long long)t.addr);
        return &(images_[t.addr] = img);
    }

    void DestroyImage(Image& img) {
        if (img.image) vkDestroyImage(device_, img.image, nullptr);
        if (img.memory) vkFreeMemory(device_, img.memory, nullptr);
        img = Image{};
    }

    // ---- commands -------------------------------------------------------------------------------------------
    void BeginCommands() {
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &bi);
    }

    void SubmitAndWait(VkSemaphore wait, VkSemaphore signal) {
        vkEndCommandBuffer(cmd_);
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        if (wait) {
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &wait;
            si.pWaitDstStageMask = &stage;
        }
        if (signal) {
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &signal;
        }
        VkResult r = vkQueueSubmit(queue_, 1, &si, fence_);
        if (r != VK_SUCCESS) {
            LogOnce("submit", "vkQueueSubmit failed: %d", r);
            return;
        }
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, 5000000000ull);
        vkResetFences(device_, 1, &fence_);
    }

    void Barrier(VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {aspect, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
    }
    void Transition(Image& img, VkImageLayout to) {
        if (img.layout == to) return;
        Barrier(img.image, img.aspect, img.layout, to);
        img.layout = to;
    }

    void ClearColorImage(Image& img, const uint32_t color[4], uint8_t kind) {
        if (IsDepthFormat(img.nvn_format)) return;
        Transition(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue v;
        memcpy(v.uint32, color, sizeof(v.uint32));  // float/int/uint share the bits
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd_, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
        img.written = true;
        if (kind == kClearFloat && clear_colors_logged_ < 16) {
            float f[4];
            memcpy(f, color, sizeof(f));
            uint64_t key = (uint64_t(img.nvn_format) << 48) ^ (uint64_t(color[0]) * 31 + color[1]) * 31 + color[2];
            if (logged_clears_.insert(key).second) {
                clear_colors_logged_++;
                LOGI(kTag, "clear %dx%d (format %d) to (%.3f, %.3f, %.3f, %.3f)", img.width, img.height,
                     img.nvn_format, f[0], f[1], f[2], f[3]);
            }
        }
    }

    void CopyImage(Image& src, const CopyRegion& sr, Image& dst, const CopyRegion& dr, int flags) {
        if (IsDepthFormat(src.nvn_format) != IsDepthFormat(dst.nvn_format)) return;
        auto clampw = [](int v, int def) { return v > 0 ? v : def; };
        int sw = clampw(sr.width, src.width), sh = clampw(sr.height, src.height);
        int dw = clampw(dr.width, dst.width), dh = clampw(dr.height, dst.height);
        Transition(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        Transition(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const bool can_blit = (src.features & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
                              (dst.features & VK_FORMAT_FEATURE_BLIT_DST_BIT) &&
                              (!IsDepthFormat(src.nvn_format) || src.vk_format == dst.vk_format);
        if (can_blit) {
            VkImageBlit b{};
            b.srcSubresource = {src.aspect, 0, 0, 1};
            b.srcOffsets[0] = {sr.x, sr.y, 0};
            b.srcOffsets[1] = {sr.x + sw, sr.y + sh, 1};
            b.dstSubresource = {dst.aspect, 0, 0, 1};
            b.dstOffsets[0] = {dr.x, dr.y, 0};
            b.dstOffsets[1] = {dr.x + dw, dr.y + dh, 1};
            if (flags & 0x8) std::swap(b.dstOffsets[0].y, b.dstOffsets[1].y);  // MIRROR_Y
            if (flags & 0x4) std::swap(b.dstOffsets[0].x, b.dstOffsets[1].x);  // MIRROR_X
            VkFilter filter = ((flags & 0x1) && !IsDepthFormat(src.nvn_format) &&
                               (src.features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
                                  ? VK_FILTER_LINEAR
                                  : VK_FILTER_NEAREST;
            vkCmdBlitImage(cmd_, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, filter);
        } else if (src.vk_format == dst.vk_format && sw == dw && sh == dh) {
            VkImageCopy c{};
            c.srcSubresource = {src.aspect, 0, 0, 1};
            c.srcOffset = {sr.x, sr.y, 0};
            c.dstSubresource = {dst.aspect, 0, 0, 1};
            c.dstOffset = {dr.x, dr.y, 0};
            c.extent = {uint32_t(sw), uint32_t(sh), 1};
            vkCmdCopyImage(cmd_, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        } else {
            LogOnce("copy", "cannot copy format %d -> %d yet", src.nvn_format, dst.nvn_format);
            return;
        }
        dst.written = dst.written || src.written;
        if (copies_logged_ < 16) {
            copies_logged_++;
            LOGI(kTag, "copy %dx%d (format %d) -> %dx%d (format %d), flags 0x%x", sw, sh, src.nvn_format, dw, dh,
                 dst.nvn_format, flags);
        }
    }

    void LogTargets(const std::vector<TexRef>& rts, int num_rt) {
        if (target_sets_logged_ >= 24) return;
        std::string desc;
        uint64_t key = 1469598103934665603ull;
        for (size_t i = 0; i < rts.size(); i++) {
            const TexRef& t = rts[i];
            if (!t.valid) continue;
            char buf[64];
            snprintf(buf, sizeof(buf), "%s%s%dx%d fmt %d", desc.empty() ? "" : ", ",
                     int(i) == num_rt ? "depth " : "", t.tex.width, t.tex.height, t.tex.format);
            desc += buf;
            key = (key ^ (t.addr + uint64_t(t.tex.format) * 7919 + i)) * 1099511628211ull;
        }
        if (desc.empty() || !logged_targets_.insert(key).second) return;
        target_sets_logged_++;
        LOGI(kTag, "render targets: %s", desc.c_str());
    }

    template <typename... A>
    void LogOnce(const char* key, const char* fmt, A... args) {
        if (!logged_once_.insert(key).second) return;
        LOGW(kTag, fmt, args...);
    }

    std::mutex m_;
    bool init_failed_ = false;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props_{};
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t family_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkSemaphore acquire_sem_ = VK_NULL_HANDLE, render_sem_ = VK_NULL_HANDLE;

    void* window_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    bool swapchain_dirty_ = false;
    VkExtent2D sc_extent_{};
    std::vector<VkImage> sc_images_;

    std::unordered_map<uint64_t, Image> images_;
    std::unordered_map<int, std::pair<VkFormat, VkFormatFeatureFlags>> format_cache_;

    // diagnostics
    uint64_t presented_ = 0;
    int images_created_ = 0, clear_colors_logged_ = 0, copies_logged_ = 0, target_sets_logged_ = 0;
    bool logged_first_image_ = false;
    std::set<uint64_t> logged_clears_, logged_targets_;
    std::set<std::string> logged_once_;
};

}  // namespace

Backend* CreatePlatformBackend() { return new VulkanBackend; }

}  // namespace dsr::hle::nvn

#endif  // __ANDROID__
