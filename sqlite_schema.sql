
CREATE TABLE usuarios (
    id SERIAL PRIMARY KEY, 
    name VARCHAR NOT NULL, 
    username VARCHAR, 
    hashed_password VARCHAR, 
    role VARCHAR
);

CREATE UNIQUE INDEX ix_usuarios_username ON usuarios (username);
CREATE INDEX ix_usuarios_id ON usuarios (id);

CREATE TABLE dado_ia (
    id SERIAL PRIMARY KEY, 
    documento_id VARCHAR, 
    data_criacao TIMESTAMP WITHOUT TIME ZONE, 
    responsavel VARCHAR NOT NULL, 
    data_finalizacao TIMESTAMP WITHOUT TIME ZONE, 
    responsavel_assistencia VARCHAR, 
    status VARCHAR, 
    produto VARCHAR, 
    quantidade INTEGER, 
    observacao_producao VARCHAR, 
    falha VARCHAR, 
    observacao_assistencia VARCHAR, 
    localizacao_componente VARCHAR, 
    lado_placa VARCHAR, 
    setor VARCHAR, 
    observacao VARCHAR, 
    resultado_ia VARCHAR, 
    falhas_json JSONB
);

CREATE INDEX ix_dado_ia_data_criacao ON dado_ia (data_criacao);
CREATE INDEX ix_dado_ia_status ON dado_ia (status);
CREATE UNIQUE INDEX ix_dado_ia_documento_id ON dado_ia (documento_id);
CREATE INDEX ix_dado_ia_id ON dado_ia (id);

CREATE TABLE registros_producao (
    id SERIAL PRIMARY KEY, 
    data_registro DATE NOT NULL, 
    tipo_registro VARCHAR(1) NOT NULL, 
    quantidade_diaria INTEGER, 
    quantidade_mensal INTEGER NOT NULL, 
    observacao_mensal VARCHAR, 
    observacao_diaria VARCHAR, 
    responsavel VARCHAR NOT NULL, 
    CONSTRAINT uq_data_tipo_registro UNIQUE (data_registro, tipo_registro)
);

CREATE INDEX ix_registros_producao_id ON registros_producao (id);
CREATE INDEX ix_registros_producao_data_registro ON registros_producao (data_registro);