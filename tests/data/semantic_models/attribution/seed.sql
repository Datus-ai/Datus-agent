CREATE TABLE ledger (
    d DATE,
    cat VARCHAR,
    x DOUBLE,
    y DOUBLE
);

INSERT INTO ledger VALUES
    -- Jan/Feb: term_wise, mix_shift, and factor_shapley.
    (DATE '2024-01-05', 'a', 100, 20),
    (DATE '2024-01-20', 'b', 50, 10),
    (DATE '2024-02-10', 'a', 80, 20),
    (DATE '2024-02-15', 'b', 30, 10),
    (DATE '2024-02-20', 'a', 120, 60),

    -- Mar/Apr: SQL NULL and the literal string '(null)' must stay distinct.
    (DATE '2024-03-05', NULL, 100, 0),
    (DATE '2024-03-05', '(null)', 40, 0),
    (DATE '2024-04-10', NULL, 130, 0),
    (DATE '2024-04-10', '(null)', 60, 0),

    -- May/Jun: the largest combined-weight member is a negative mover.
    (DATE '2024-05-05', 'shrink', 3000, 0),
    (DATE '2024-05-05', 'big', 1000, 0),
    (DATE '2024-06-10', 'shrink', 100, 0),
    (DATE '2024-06-10', 'big', 1100, 0),

    -- Nov/Dec: parameter binding changes the metric result.
    (DATE '2024-11-05', 'a', 200, 0),
    (DATE '2024-11-12', 'a', 60, 0),
    (DATE '2024-11-20', 'b', 40, 0),
    (DATE '2024-12-05', 'a', 300, 0),
    (DATE '2024-12-12', 'b', 120, 0),
    (DATE '2024-12-20', 'b', 30, 0);
