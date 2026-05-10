#include "NeuralBotWSHandler.h"
#include "NeuralBotMgr.h"
#include "Log.h"

#include <boost/asio.hpp>
#include <sstream>
#include <cstring>
#include <vector>

using boost::asio::ip::tcp;

NeuralBotWSHandler& NeuralBotWSHandler::instance()
{
    static NeuralBotWSHandler inst;
    return inst;
}

void NeuralBotWSHandler::Start(uint16 port)
{
    _port = port;
    _running = true;
    _thread = std::thread(&NeuralBotWSHandler::AcceptLoop, this);
    LOG_INFO("module.neuralbot", "WebSocket handler started on port {}", port);
}

void NeuralBotWSHandler::Stop()
{
    _running = false;
    if (_ioContext)
        _ioContext->stop();
    if (_thread.joinable())
        _thread.join();
}

void NeuralBotWSHandler::AcceptLoop()
{
    _ioContext = std::make_shared<boost::asio::io_context>();
    tcp::acceptor acceptor(*_ioContext, tcp::endpoint(tcp::v4(), _port));

    while (_running)
    {
        try
        {
            tcp::socket socket(*_ioContext);
            boost::system::error_code ec;
            acceptor.accept(socket, ec);
            if (ec)
                continue;
            HandleClient(std::move(socket));
        }
        catch (...)
        {
            if (!_running)
                break;
        }
    }
}

void NeuralBotWSHandler::HandleClient(tcp::socket socket)
{
    try
    {
        boost::asio::streambuf buf;
        while (_running && socket.is_open())
        {
            boost::system::error_code ec;
            size_t len = boost::asio::read_until(socket, buf, '\n', ec);
            if (ec)
                break;

            std::istream is(&buf);
            std::string line;
            std::getline(is, line);

            if (line.empty())
                continue;

            std::string response = ProcessMessage(line);
            response += "\n";
            boost::asio::write(socket, boost::asio::buffer(response), ec);
            if (ec)
                break;
        }
    }
    catch (...)
    {
    }
}

std::string NeuralBotWSHandler::ProcessMessage(const std::string& msg)
{
    if (msg.substr(0, 5) == "STEP ")
    {
        uint32 action = ACTION_NOOP;
        try { action = std::stoul(msg.substr(5)); } catch (...) {}

        NeuralBotStepResult result = sNeuralBotMgr.Step(action);

        std::ostringstream os;
        os << "RESULT";
        os << " " << (result.done ? "1" : "0");
        os << " " << result.reward.total;

        float flat[OBS_TOTAL_SIZE];
        result.observation.ToFloatArray(flat);
        for (size_t i = 0; i < OBS_TOTAL_SIZE; ++i)
            os << " " << flat[i];

        // Append reward components for analysis
        NeuralBotReward const& r = result.reward;
        os << " " << r.xpDelta;
        os << " " << r.damageTaken;
        os << " " << r.killReward;
        os << " " << r.deathPenalty;
        os << " " << r.lootReward;
        os << " " << r.questAccepted;
        os << " " << r.questCompleted;
        os << " " << r.questProximity;
        os << " " << r.questProgress;
        os << " " << r.timePenalty;

        return os.str();
    }
    else if (msg == "RESET")
    {
        NeuralBotObservation obs = sNeuralBotMgr.Reset();

        std::ostringstream os;
        os << "OBS";
        float flat[OBS_TOTAL_SIZE];
        obs.ToFloatArray(flat);
        for (size_t i = 0; i < OBS_TOTAL_SIZE; ++i)
            os << " " << flat[i];

        return os.str();
    }
    else if (msg == "PING")
    {
        return "PONG";
    }
    else if (msg.substr(0, 10) == "SET_SPELLS ")
    {
        std::istringstream is(msg.substr(10));
        for (size_t i = 0; i < 5; ++i)
        {
            uint32 spellId = 0;
            is >> spellId;
            sNeuralBotMgr.SetSpellSlot(i, spellId);
        }
        return "OK";
    }
    else if (msg == "STATUS")
    {
        Player* bot = sNeuralBotMgr.GetBotPlayer();
        std::ostringstream os;
        os << "STATUS";
        os << " " << (bot ? "READY" : "NO_BOT");
        if (bot)
        {
            os << " " << bot->GetName();
            os << " " << static_cast<int>(bot->GetLevel());
            os << " " << bot->GetZoneId();
        }
        return os.str();
    }
    else if (msg == "SPELLS")
    {
        std::vector<uint32> spells;
        sNeuralBotMgr.GetSpellbook(spells);
        std::ostringstream os;
        os << "SPELLS";
        for (size_t i = 0; i < spells.size(); ++i)
            os << " " << spells[i];
        return os.str();
    }
    else if (msg.substr(0, 13) == "SEND_SPELLBOOK ")
    {
        std::istringstream is(msg.substr(13));
        std::vector<uint32> spells;
        uint32 spellId = 0;
        while (is >> spellId)
            spells.push_back(spellId);
        sNeuralBotMgr.SetSpellSlots(spells);
        return "OK";
    }

    return "ERR unknown command";
}

bool NeuralBotWSHandler::HasPendingAction() const
{
    return _actionReady;
}

uint32 NeuralBotWSHandler::GetPendingAction()
{
    std::lock_guard<std::mutex> lock(_actionMutex);
    return _pendingAction;
}

void NeuralBotWSHandler::ClearPendingAction()
{
    std::lock_guard<std::mutex> lock(_actionMutex);
    _actionReady = false;
}

void NeuralBotWSHandler::SendStepResult(const NeuralBotStepResult& result)
{
}

void NeuralBotWSHandler::SendResetResult(const NeuralBotObservation& obs)
{
}

void NeuralBotWSHandler::SendError(const std::string& msg)
{
}
