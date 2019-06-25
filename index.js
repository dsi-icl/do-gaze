const express = require('express');
const app = express();
const expressWs = require('express-ws')(app);
const port = 3000;
const aWss = expressWs.getWss('/');

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/', function (req, res, next) {
    res.sendfile(__dirname + '/ws.html');
});

app.ws('/', function (ws, req) {
    ws.on('message', function (msg) {
        console.log(msg);
        aWss.clients.forEach(function (client) {
            client.send(JSON.stringify({
                date: (new Date()).getTime(),
                message: msg,
            }));
        });
    });
    console.log('socket', req.testing);
});

app.listen(port, function () {
    console.log("Server is running on " + port + " port");
});
