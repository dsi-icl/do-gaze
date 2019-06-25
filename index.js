const express = require('express');
const app = express();
const expressWs = require('express-ws')(app);
const port = 3000;
const aWss = expressWs.getWss('/');

const broadcast = function (data) {
    aWss.clients.forEach(function (client) {
        client.send(JSON.stringify({
            timestamp: (new Date()).getTime(),
            ...data
        }));
    });
}

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/', function (req, res, next) {
    res.sendFile(__dirname + '/ws.html');
});

app.post('/', function (req, res, next) {
    broadcast(req.body);
    res.json({
        status: "OK"
    });
});

app.ws('/', function (ws, req) {
    ws.on('message', function (msg) {
        console.log(msg);
        broadcast({
            message: msg,
        })
    });
    console.log('socket', req.testing);
});

app.listen(port, function () {
    console.log("Server is running on " + port + " port");
});
