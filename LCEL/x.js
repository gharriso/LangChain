// Connect to MySQL Database
const mysql = require('mysql');

const connection = mysql.createConnection({
    host     : 'localhost',
    user     : 'root',
    password : 'password',
    database : 'sakila'
});
async function main() {
    try {
        await connection.connect();
        // Write an sql query that returns the names of all the films in the database
        const query = 'SELECT title FROM film';
        const results= await connection.query(query);
        console.log(results);
 
        console.log('The names of all the films in the database are: ', results);
        console.log('connected as id ' + connection.threadId);
    } catch (err) {
        console.error('error connecting: ' + err.stack);
    }
}

main();