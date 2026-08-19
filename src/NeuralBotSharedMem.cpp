#include "NeuralBotSharedMem.h"
#include "Log.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/eventfd.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdint>

NeuralBotSharedMem& NeuralBotSharedMem::instance()
{
    static NeuralBotSharedMem inst;
    return inst;
}

bool NeuralBotSharedMem::Create(uint32_t numBots)
{
    _name = "/neuralbot_shm";

    int fd = shm_open(_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0)
    {
        LOG_ERROR("module.neuralbot", "shm_open failed: {} ({})", errno, strerror(errno));
        return false;
    }

    if (ftruncate(fd, SHM_TOTAL_SIZE) < 0)
    {
        LOG_ERROR("module.neuralbot", "ftruncate failed: {} ({})", errno, strerror(errno));
        close(fd);
        shm_unlink(_name.c_str());
        return false;
    }

    _ptr = static_cast<uint8_t*>(mmap(nullptr, SHM_TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);

    if (_ptr == MAP_FAILED)
    {
        LOG_ERROR("module.neuralbot", "mmap failed: {} ({})", errno, strerror(errno));
        _ptr = nullptr;
        shm_unlink(_name.c_str());
        return false;
    }

    NeuralBotSharedControl* ctrl = GetControl();
    ctrl->magic      = SHM_MAGIC;
    ctrl->version    = SHM_VERSION;
    ctrl->num_bots   = numBots;
    ctrl->step_count = 0;
    ctrl->eventfd    = eventfd(0, EFD_SEMAPHORE);
    if (ctrl->eventfd < 0)
        ctrl->eventfd = -1;
    ctrl->actions_ready = 0;
    ctrl->obs_ready     = 0;
    ctrl->shutdown      = 0;
    ctrl->frame_bytes   = SHM_FRAME_BYTES;

    std::memset(GetActionPtr(), 0, SHM_ACTIONS_SIZE);
    std::memset(GetFramesPtr(), 0, SHM_FRAME_REGION_SIZE);
    std::memset(GetDonesPtr(),  0, SHM_DONES_SIZE);

    LOG_INFO("module.neuralbot",
        "Shared memory created: {} ({} KB, {} bots, frame_bytes={})",
        _name, SHM_TOTAL_SIZE / 1024, numBots, SHM_FRAME_BYTES);
    return true;
}

void NeuralBotSharedMem::Destroy()
{
    SignalShutdown();
    if (_ptr)
    {
        NeuralBotSharedControl* ctrl = GetControl();
        if (ctrl && ctrl->eventfd >= 0)
            close(ctrl->eventfd);
        munmap(_ptr, SHM_TOTAL_SIZE);
        _ptr = nullptr;
    }
    shm_unlink(_name.c_str());
}

bool NeuralBotSharedMem::TryReadActions(uint8_t* outActions, size_t numBots)
{
    if (!_ptr) return false;

    NeuralBotSharedControl* ctrl = GetControl();
    if (ctrl->actions_ready != 1)
        return false;

    std::memcpy(outActions, GetActionPtr(), numBots * sizeof(uint8_t));
    ctrl->actions_ready = 0; // clear flag after reading
    return true;
}

bool NeuralBotSharedMem::ObservationsPending() const
{
    if (!_ptr) return false;
    return GetControl()->obs_ready == 1;
}

void NeuralBotSharedMem::WriteFrames(const uint8_t* frames, const uint8_t* dones, size_t numBots)
{
    if (!_ptr) return;

    std::memcpy(GetFramesPtr(), frames, numBots * SHM_FRAME_BYTES);
    std::memcpy(GetDonesPtr(),  dones,  numBots * sizeof(uint8_t));
}

void NeuralBotSharedMem::SignalObservationsReady()
{
    if (!_ptr) return;

    NeuralBotSharedControl* ctrl = GetControl();
    ctrl->step_count++;
    ctrl->obs_ready = 1;

    if (ctrl->eventfd >= 0)
    {
        uint64_t val = 1;
        write(ctrl->eventfd, &val, sizeof(val));
    }
}

void NeuralBotSharedMem::SignalShutdown()
{
    if (!_ptr) return;

    NeuralBotSharedControl* ctrl = GetControl();
    ctrl->shutdown = 1;

    if (ctrl->eventfd >= 0)
    {
        uint64_t val = 1;
        write(ctrl->eventfd, &val, sizeof(val));
    }
}

bool NeuralBotSharedMem::IsShutdown() const
{
    if (!_ptr) return true;
    return GetControl()->shutdown != 0;
}
