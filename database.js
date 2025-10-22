const mysql = require('mysql');

// Konfigurasi koneksi database
const koneksi = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: '',
    database: 'info_pasca',
    multipleStatements: true
});

// Koneksi database
koneksi.connect((err) => {
    if (err) {
        console.error('❌ MySQL Connection Error:', err);
        throw err;
    }
    console.log('✅ MySQL Connected to database: info_pasca');
});

// Handle connection lost
koneksi.on('error', function(err) {
    console.error('❌ Database error:', err);
    if(err.code === 'PROTOCOL_CONNECTION_LOST') {
        console.log('🔄 Reconnecting to database...');
    } else {
        throw err;
    }
});

module.exports = koneksi;