-- Scored items in two products, one item per day. Ranks are by total score,
-- descending, within the product:
--   p1: e=40 (1), d=30 (2), b=20 (3), c=20 (3), a=10 (5)   -> 5 items
--   p2: j=9 (1),  i=7 (2),  f=5 (3), g=5 (3), h=5 (3)      -> 5 items
CREATE TABLE activity_scores (
    product VARCHAR NOT NULL,
    item    VARCHAR NOT NULL,
    day     DATE NOT NULL,
    score   DOUBLE NOT NULL
);

INSERT INTO activity_scores (product, item, day, score) VALUES
    ('p1', 'a', DATE '2025-06-01', 10),
    ('p1', 'b', DATE '2025-06-02', 20),
    ('p1', 'c', DATE '2025-06-03', 20),
    ('p1', 'd', DATE '2025-06-04', 30),
    ('p1', 'e', DATE '2025-06-05', 40),
    ('p2', 'f', DATE '2025-06-01', 5),
    ('p2', 'g', DATE '2025-06-02', 5),
    ('p2', 'h', DATE '2025-06-03', 5),
    ('p2', 'i', DATE '2025-06-04', 7),
    ('p2', 'j', DATE '2025-06-05', 9);
