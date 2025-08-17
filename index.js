const fs = require('fs');
const path = require('path');
const { Parser } = require('json2csv');

// Pega a fórmula via argumento de linha de comando
const formula = process.argv[2];

if (!formula) {
    console.error('Por favor, forneça a fórmula como argumento. Exemplo: node index.js "x * 2"');
    process.exit(1);
}
const results = [];

// Loop through values from 1 to 1000
for (let i = 0; i < 100000; i++) {
    const x = Math.floor(Math.random() * 100) + 1;
    const y = eval(formula.replace(/x/g, x));
    results.push({ x, y });
}

// Define the path for thye CSV file
const csvFilePath = path.join(__dirname, './data.csv');

// Convert results to CSV format
const json2csvParser = new Parser();
const csv = json2csvParser.parse(results);

// Write the CSV file
fs.writeFile(csvFilePath, csv, (err) => {
    if (err) {
        console.error('Error writing to CSV file', err);
    } else {
        console.log('Arquivo CSV foi criado com sucesso.');
    }
});