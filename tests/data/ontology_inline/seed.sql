-- Seed for the ossie-mini-flights scenario. Four airports, eight flights,
-- sized so both metrics can be checked by eye and, more to the point, so the
-- two of them DISAGREE on every airport. If lowering drops the correlation
-- path the two metrics collapse into one and this table is how you find out.
--
--   flight  origin -> dest   dep_delay
--   F1      ATL -> DCA        10
--   F2      ATL -> DCA        22
--   F3      DCA -> ATL         0
--   F4      SFO -> ATL        60
--   F5      ORD -> SFO        30
--   F6      SFO -> ORD         4
--   F7      DCA -> SFO        40
--   F8      ATL -> ORD        16
--
-- airport_average_departure_delay        -- group by flight_departs_from.code
--   ATL 16 (n=3: 10+22+16=48)   DCA 20 (n=2: 0+40)
--   SFO 32 (n=2: 60+4)          ORD 30 (n=1: 30)
--
-- airport_average_inbound_departure_delay -- group by flight_arrives_at.code
--   ATL 30 (n=2: 0+60)          DCA 16 (n=2: 10+22)
--   SFO 35 (n=2: 30+40)         ORD 10 (n=2: 4+16)
--
-- All four airports differ between the two, and every average is an integer.
-- Grouping either metric by a bare AIRPORT.code is `ambiguous_join_path` --
-- that refusal is part of what this fixture demonstrates, not a defect.

CREATE TABLE airports (
  code VARCHAR NOT NULL
);

CREATE TABLE flights (
  id          VARCHAR NOT NULL,
  origin_code VARCHAR NOT NULL,
  dest_code   VARCHAR NOT NULL,
  dep_delay   INTEGER NOT NULL
);

INSERT INTO airports (code) VALUES ('ATL'), ('DCA'), ('SFO'), ('ORD');

INSERT INTO flights (id, origin_code, dest_code, dep_delay) VALUES
  ('F1', 'ATL', 'DCA', 10),
  ('F2', 'ATL', 'DCA', 22),
  ('F3', 'DCA', 'ATL',  0),
  ('F4', 'SFO', 'ATL', 60),
  ('F5', 'ORD', 'SFO', 30),
  ('F6', 'SFO', 'ORD',  4),
  ('F7', 'DCA', 'SFO', 40),
  ('F8', 'ATL', 'ORD', 16);
