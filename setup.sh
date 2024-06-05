#!/bin/bash

echo "Setting up PostgreSQL database..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASSWORD';
    ALTER ROLE $DB_USER SUPERUSER;
    CREATE DATABASE $DB_NAME;
EOSQL

echo "Database setup complete."
