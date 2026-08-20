#ifndef NEURALBOTWSHANDLER_H
#define NEURALBOTWSHANDLER_H

#include "NeuralBotCommon.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <functional>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;

class NeuralBotWSHandler
{
public:
    static NeuralBotWSHandler& instance();

    void Start(uint16 port);
    void Stop();
    bool IsRunning() const { return _running; }

    bool HasPendingAction() const;
    uint32 GetPendingAction();
    void ClearPendingAction();

    void SendStepResult(const NeuralBotStepResult& result);
    void SendResetResult(const NeuralBotObservation& obs);
    void SendError(const std::string& msg);

    void SetOnStepCallback(std::function<void(uint32)> cb) { _onStep = cb; }
    void SetOnResetCallback(std::function<void()> cb) { _onReset = cb; }

private:
    NeuralBotWSHandler() = default;
    void AcceptLoop();
    void HandleClient(tcp::socket socket);
    std::string ProcessMessage(const std::string& msg);

    std::atomic<bool> _running{false};
    std::atomic<bool> _actionReady{false};
    uint32 _pendingAction = ACTION_NOOP;
    std::mutex _actionMutex;

    std::function<void(uint32)> _onStep;
    std::function<void()> _onReset;

    uint16 _port = 9000;
    std::thread _thread;
    std::shared_ptr<boost::asio::io_context> _ioContext;
    std::shared_ptr<tcp::acceptor> _acceptor;

public:
    ~NeuralBotWSHandler()
    {
        // Safety net if Stop() was never called (static destructor at exit):
        // detach instead of aborting on a joinable thread.
        if (_thread.joinable())
            _thread.detach();
    }
};

#define sNeuralBotWS NeuralBotWSHandler::instance()

#endif
