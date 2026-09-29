const express = require("express");

const app = express();

app.use(express.json());

let produtos = [
    { id: 1, nome: "Notebook", preco: 3500 },
    { id: 2, nome: "Mouse", preco: 100 }
];

app.get("/fecaf/produtos", (req, res) => {
    res.json(produtos);
});

app.get("/fecaf/produtos/nome/:nome", (req, res) => {
    const nome = req.params.nome.toLowerCase();

    const resultado = produtos.filter(produto =>
        produto.nome.toLowerCase().includes(nome)
    );

    res.json(resultado);
});

app.get("/fecaf/produtos/:id", (req, res) => {
    const id = Number(req.params.id);
    const produto = produtos.find(produto => produto.id === id);

    if (!produto) {
        return res.status(404).json({ mensagem: "Produto não encontrado" });
    }

    res.json(produto);
});

app.post("/fecaf/produtos", (req, res) => {
    const novoProduto = {
        id: produtos.length > 0
            ? Math.max(...produtos.map(produto => produto.id)) + 1
            : 1,
        nome: req.body.nome,
        preco: req.body.preco
    };

    produtos.push(novoProduto);

    res.status(201).json(novoProduto);
});

app.put("/fecaf/produtos/:id", (req, res) => {
    const id = Number(req.params.id);
    const produto = produtos.find(produto => produto.id === id);

    if (!produto) {
        return res.status(404).json({ mensagem: "Produto não encontrado" });
    }

    produto.nome = req.body.nome;
    produto.preco = req.body.preco;

    res.json(produto);
});

app.delete("/fecaf/produtos/:id", (req, res) => {
    const id = Number(req.params.id);
    const index = produtos.findIndex(produto => produto.id === id);

    if (index === -1) {
        return res.status(404).json({ mensagem: "Produto não encontrado" });
    }

    produtos.splice(index, 1);

    res.json({ mensagem: "Produto excluído com sucesso" });
});

app.listen(3000, () => {
    console.log("Servidor rodando na porta 3000");
});