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
    LOG_INFO("module.neuralbot", "NeuralBot TCP handler started on port {}", port);
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
            std::thread(&NeuralBotWSHandler::HandleClient, this, std::move(socket)).detach();
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
            if (!response.empty())
            {
                response += "\n";
                boost::asio::write(socket, boost::asio::buffer(response), ec);
            }
        }
    }
    catch (...)
    {
    }
}

std::string NeuralBotWSHandler::ProcessMessage(const std::string& msg)
{
    std::istringstream iss(msg);
    std::string cmd;
    iss >> cmd;

    if (cmd == "PING")
        return "PONG";

    if (cmd == "STEP")
    {
        std::string botName;
        int action = 0;
        iss >> botName >> action;

        NeuralBotStepResult result = sNeuralBotMgr.Step(botName, static_cast<uint32>(action));

        std::ostringstream os;
        os << "RESULT";
        os << " " << (result.done ? "1" : "0");
        os << " " << result.reward.total;

        float flat[OBS_TOTAL_SIZE];
        result.observation.ToFloatArray(flat);
        for (size_t i = 0; i < OBS_TOTAL_SIZE; ++i)
            os << " " << flat[i];

        NeuralBotReward const& r = result.reward;
        os << " " << r.xpDelta << " " << r.damageTaken << " " << r.killReward
           << " " << r.deathPenalty << " " << r.lootReward
           << " " << r.questAccepted << " " << r.questCompleted
           << " " << r.questProximity << " " << r.questProgress
           << " " << r.timePenalty;

        return os.str();
    }

    if (cmd == "RESET")
    {
        std::string botName;
        iss >> botName;

        NeuralBotObservation obs = sNeuralBotMgr.Reset(botName);
        std::ostringstream os;
        os << "OBS";
        float flat[OBS_TOTAL_SIZE];
        obs.ToFloatArray(flat);
        for (size_t i = 0; i < OBS_TOTAL_SIZE; ++i)
            os << " " << flat[i];
        return os.str();
    }

    if (cmd == "STATUS")
    {
        std::string botName;
        iss >> botName;

        NeuralBotInstance* inst = sNeuralBotMgr.GetInstance(botName);
        std::ostringstream os;
        os << "STATUS";
        if (inst && inst->GetPlayer())
        {
            Player* p = inst->GetPlayer();
            os << " READY " << p->GetName() << " " << uint32(p->GetLevel()) << " " << p->GetZoneId();
        }
        else
            os << " NO_BOT";
        return os.str();
    }

    if (cmd == "BOTS")
    {
        auto names = sNeuralBotMgr.GetBotNames();
        std::ostringstream os;
        os << "BOTS";
        for (auto const& n : names)
            os << " " << n;
        return os.str();
    }

    if (cmd == "SPELLS")
    {
        std::string botName;
        iss >> botName;

        NeuralBotInstance* inst = sNeuralBotMgr.GetInstance(botName);
        if (!inst)
            return "ERR bot not found";

        std::vector<uint32> spells;
        inst->GetSpellbook(spells);
        std::ostringstream os;
        os << "SPELLS";
        for (uint32 s : spells)
            os << " " << s;
        return os.str();
    }

    if (cmd == "SET_SPELLS")
    {
        std::string botName;
        iss >> botName;

        NeuralBotInstance* inst = sNeuralBotMgr.GetInstance(botName);
        if (!inst)
            return "ERR bot not found";

        std::vector<uint32> spellIds;
        uint32 s;
        while (iss >> s)
            spellIds.push_back(s);
        inst->SetSpellSlots(spellIds);
        return "OK";
    }

    return "ERR unknown command";
}
