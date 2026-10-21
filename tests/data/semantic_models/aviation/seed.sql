-- Seed for the aviation scenario: a reconstruction of Palantir's public
-- Aviation reference ontology (Flight / Aircraft / Carrier / Airport / Route /
-- FlightSensor / FlightAlert / Delay) sized so every expected row is checkable
-- by eye. Benchmark clock: 2026-08-15 12:00 UTC.
--
-- Deliberate shapes (see design/palantir-ontology-query-coverage.zh.md):
--   * airports is reached from flights by two roles (origin / destination)
--     and again through routes — same target, several paths.
--   * delays and flight_alerts are two independent 1:N children of flights;
--     a query that joins both at once multiplies passengers (fan-out probe).
--   * sensor_readings is a narrow time series two hops below flights.
--   * F1013 (DL503) is the one flight whose direct origin (LAS) disagrees with
--     its route (R-SEA-SFO): the consistency case must return exactly it.
--     It has no delays, alerts or sensors, so no other expectation moves.
--
-- Hand tallies used by queries.yaml / select-cases.yaml:
--   delay minutes by carrier: SK 12+55+42+65+30+60=264, UA 30+25+30=85, DL 0
--   delay minutes by aircraft: N200SK 55+65+30=150, N100SK 12+42+60=114,
--                              N300UA 25+30=55, N400UA 30
--   alerts on SFO-origin flights by aircraft: N100SK 4 (A003 A004 A009 A010),
--     N200SK 3, N300UA 1, N400UA 1 — N400UA's A002 is on a JFK-origin flight
--   delayed flights by destination: LAX 3 (F1001 F1006 F1011), JFK 2, SFO 2, SEA 1
--   avg delay by route: R-SFO-JFK (55+65+30)/3=50, R-SFO-LAX (12+42+60)/3=38,
--     R-JFK-SFO 30, R-LAS-SFO 30, R-SFO-SEA 25
--   open alerts: A002 A004 A005 A006 A007 A009 A010 A011 (8)
--   flights with an open alert by carrier: SK 3/5 (F1006 F1008 F1011),
--     UA 3/5 (F1004 F1007 F1012), DL 0/3
CREATE TABLE airports (
    airport_code VARCHAR,
    name VARCHAR,
    city VARCHAR,
    timezone VARCHAR
);
INSERT INTO airports VALUES
    ('SFO', 'San Francisco International Airport', 'San Francisco', 'America/Los_Angeles'),
    ('LAX', 'Los Angeles International Airport', 'Los Angeles', 'America/Los_Angeles'),
    ('JFK', 'John F. Kennedy International Airport', 'New York', 'America/New_York'),
    ('SEA', 'Seattle-Tacoma International Airport', 'Seattle', 'America/Los_Angeles'),
    ('LAS', 'Harry Reid International Airport', 'Las Vegas', 'America/Los_Angeles');

CREATE TABLE carriers (
    carrier_id VARCHAR,
    iata_code VARCHAR,
    name VARCHAR
);
INSERT INTO carriers VALUES
    ('C-SK', 'SK', 'Sky Industries Air'),
    ('C-UA', 'UA', 'United Example'),
    ('C-DL', 'DL', 'Delta Example');

CREATE TABLE aircraft (
    aircraft_id VARCHAR,
    tail_number VARCHAR,
    model VARCHAR,
    carrier_id VARCHAR,
    in_service BOOLEAN
);
INSERT INTO aircraft VALUES
    ('AC-001', 'N100SK', 'A320', 'C-SK', TRUE),
    ('AC-002', 'N200SK', 'A321', 'C-SK', TRUE),
    ('AC-003', 'N300UA', 'B737-900', 'C-UA', TRUE),
    ('AC-004', 'N400UA', 'B787-9', 'C-UA', TRUE),
    ('AC-005', 'N500DL', 'A220', 'C-DL', TRUE);

CREATE TABLE routes (
    route_id VARCHAR,
    origin_airport_code VARCHAR,
    destination_airport_code VARCHAR,
    distance_miles INTEGER
);
INSERT INTO routes VALUES
    ('R-SFO-LAX', 'SFO', 'LAX', 337),
    ('R-LAX-SFO', 'LAX', 'SFO', 337),
    ('R-SFO-JFK', 'SFO', 'JFK', 2586),
    ('R-JFK-SFO', 'JFK', 'SFO', 2586),
    ('R-SEA-SFO', 'SEA', 'SFO', 679),
    ('R-SFO-SEA', 'SFO', 'SEA', 679),
    ('R-SFO-LAS', 'SFO', 'LAS', 414),
    ('R-LAS-SFO', 'LAS', 'SFO', 414);

CREATE TABLE flights (
    flight_id VARCHAR,
    flight_number VARCHAR,
    carrier_id VARCHAR,
    aircraft_id VARCHAR,
    route_id VARCHAR,
    origin_airport_code VARCHAR,
    destination_airport_code VARCHAR,
    scheduled_departure TIMESTAMP,
    actual_departure TIMESTAMP,
    scheduled_arrival TIMESTAMP,
    actual_arrival TIMESTAMP,
    passenger_count INTEGER,
    status VARCHAR
);
INSERT INTO flights VALUES
    ('F1001', 'SK101', 'C-SK', 'AC-001', 'R-SFO-LAX', 'SFO', 'LAX', TIMESTAMP '2026-08-08 08:00:00', TIMESTAMP '2026-08-08 08:10:00', TIMESTAMP '2026-08-08 09:30:00', TIMESTAMP '2026-08-08 09:42:00', 152, 'COMPLETED'),
    ('F1002', 'SK201', 'C-SK', 'AC-002', 'R-SFO-JFK', 'SFO', 'JFK', TIMESTAMP '2026-08-09 07:00:00', TIMESTAMP '2026-08-09 07:45:00', TIMESTAMP '2026-08-09 15:30:00', TIMESTAMP '2026-08-09 16:25:00', 188, 'COMPLETED'),
    ('F1003', 'UA301', 'C-UA', 'AC-003', 'R-LAX-SFO', 'LAX', 'SFO', TIMESTAMP '2026-08-10 10:00:00', TIMESTAMP '2026-08-10 10:05:00', TIMESTAMP '2026-08-10 11:25:00', TIMESTAMP '2026-08-10 11:30:00', 171, 'COMPLETED'),
    ('F1004', 'UA401', 'C-UA', 'AC-004', 'R-JFK-SFO', 'JFK', 'SFO', TIMESTAMP '2026-08-10 13:00:00', TIMESTAMP '2026-08-10 13:20:00', TIMESTAMP '2026-08-10 16:20:00', TIMESTAMP '2026-08-10 16:50:00', 224, 'COMPLETED'),
    ('F1005', 'DL501', 'C-DL', 'AC-005', 'R-SEA-SFO', 'SEA', 'SFO', TIMESTAMP '2026-08-11 09:00:00', TIMESTAMP '2026-08-11 09:00:00', TIMESTAMP '2026-08-11 11:05:00', TIMESTAMP '2026-08-11 11:02:00', 126, 'COMPLETED'),
    ('F1006', 'SK102', 'C-SK', 'AC-001', 'R-SFO-LAX', 'SFO', 'LAX', TIMESTAMP '2026-08-12 08:00:00', TIMESTAMP '2026-08-12 08:35:00', TIMESTAMP '2026-08-12 09:30:00', TIMESTAMP '2026-08-12 10:12:00', 149, 'COMPLETED'),
    ('F1007', 'UA302', 'C-UA', 'AC-003', 'R-SFO-SEA', 'SFO', 'SEA', TIMESTAMP '2026-08-13 11:00:00', TIMESTAMP '2026-08-13 11:15:00', TIMESTAMP '2026-08-13 13:10:00', TIMESTAMP '2026-08-13 13:35:00', 168, 'COMPLETED'),
    ('F1008', 'SK202', 'C-SK', 'AC-002', 'R-SFO-JFK', 'SFO', 'JFK', TIMESTAMP '2026-08-14 06:30:00', TIMESTAMP '2026-08-14 07:40:00', TIMESTAMP '2026-08-14 15:00:00', TIMESTAMP '2026-08-14 16:35:00', 191, 'COMPLETED'),
    ('F1009', 'DL502', 'C-DL', 'AC-005', 'R-SFO-LAS', 'SFO', 'LAS', TIMESTAMP '2026-08-14 16:00:00', TIMESTAMP '2026-08-14 16:00:00', TIMESTAMP '2026-08-14 17:35:00', TIMESTAMP '2026-08-14 17:31:00', 118, 'COMPLETED'),
    ('F1010', 'UA303', 'C-UA', 'AC-003', 'R-LAS-SFO', 'LAS', 'SFO', TIMESTAMP '2026-08-15 06:00:00', TIMESTAMP '2026-08-15 06:25:00', TIMESTAMP '2026-08-15 07:35:00', TIMESTAMP '2026-08-15 08:05:00', 162, 'COMPLETED'),
    ('F1011', 'SK103', 'C-SK', 'AC-001', 'R-SFO-LAX', 'SFO', 'LAX', TIMESTAMP '2026-08-15 08:00:00', TIMESTAMP '2026-08-15 08:50:00', TIMESTAMP '2026-08-15 09:30:00', TIMESTAMP '2026-08-15 10:30:00', 154, 'COMPLETED'),
    ('F1012', 'UA402', 'C-UA', 'AC-004', 'R-SFO-JFK', 'SFO', 'JFK', TIMESTAMP '2026-08-15 09:00:00', TIMESTAMP '2026-08-15 09:08:00', TIMESTAMP '2026-08-15 17:20:00', TIMESTAMP '2026-08-15 17:35:00', 229, 'COMPLETED'),
    ('F1013', 'DL503', 'C-DL', 'AC-005', 'R-SEA-SFO', 'LAS', 'SFO', TIMESTAMP '2026-08-15 18:00:00', TIMESTAMP '2026-08-15 18:00:00', TIMESTAMP '2026-08-15 19:30:00', TIMESTAMP '2026-08-15 19:28:00', 120, 'COMPLETED');

CREATE TABLE delays (
    delay_id VARCHAR,
    flight_id VARCHAR,
    delay_type VARCHAR,
    delay_minutes INTEGER,
    cause_detail VARCHAR
);
-- D003 is 29, not 30, so UA's three delays average to an exact 28.
-- MySQL's AVG over integers returns DECIMAL(_,4), so a non-terminating
-- mean would make delay_min_max_by_carrier test float rendering across
-- engines rather than aggregate semantics.
INSERT INTO delays VALUES
    ('D001', 'F1001', 'AIR_TRAFFIC', 12, 'SFO departure flow restriction'),
    ('D002', 'F1002', 'WEATHER', 55, 'Thunderstorms near JFK'),
    ('D003', 'F1004', 'AIR_TRAFFIC', 29, 'JFK departure congestion'),
    ('D004', 'F1006', 'MAINTENANCE', 42, 'Hydraulic inspection'),
    ('D005', 'F1007', 'CREW', 25, 'Inbound crew connection'),
    ('D006', 'F1008', 'MAINTENANCE', 65, 'Unexpected avionics maintenance'),
    ('D007', 'F1008', 'WEATHER', 30, 'JFK arrival weather'),
    ('D008', 'F1010', 'AIR_TRAFFIC', 30, 'SFO arrival flow restriction'),
    ('D009', 'F1011', 'MAINTENANCE', 60, 'Brake temperature inspection');

CREATE TABLE flight_alerts (
    alert_id VARCHAR,
    flight_id VARCHAR,
    alert_type VARCHAR,
    priority VARCHAR,
    status VARCHAR,
    created_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_comment VARCHAR
);
INSERT INTO flight_alerts VALUES
    ('A001', 'F1002', 'WEATHER', 'P1', 'RESOLVED', TIMESTAMP '2026-08-09 06:40:00', TIMESTAMP '2026-08-09 07:20:00', 'Rerouted around storm cells'),
    ('A002', 'F1004', 'AIR_TRAFFIC', 'P2', 'OPEN', TIMESTAMP '2026-08-10 12:40:00', NULL, NULL),
    ('A003', 'F1006', 'MAINTENANCE', 'P1', 'RESOLVED', TIMESTAMP '2026-08-12 07:15:00', TIMESTAMP '2026-08-12 08:20:00', 'Inspection completed'),
    ('A004', 'F1006', 'OPERATIONS', 'P2', 'OPEN', TIMESTAMP '2026-08-12 08:05:00', NULL, NULL),
    ('A005', 'F1007', 'CREW', 'P2', 'OPEN', TIMESTAMP '2026-08-13 10:20:00', NULL, NULL),
    ('A006', 'F1008', 'MAINTENANCE', 'P0', 'OPEN', TIMESTAMP '2026-08-14 05:55:00', NULL, NULL),
    ('A007', 'F1008', 'WEATHER', 'P1', 'OPEN', TIMESTAMP '2026-08-14 06:10:00', NULL, NULL),
    ('A008', 'F1010', 'AIR_TRAFFIC', 'P2', 'RESOLVED', TIMESTAMP '2026-08-15 05:40:00', TIMESTAMP '2026-08-15 06:15:00', 'Flow restriction acknowledged'),
    ('A009', 'F1011', 'MAINTENANCE', 'P1', 'OPEN', TIMESTAMP '2026-08-15 07:15:00', NULL, NULL),
    ('A010', 'F1011', 'OPERATIONS', 'P2', 'OPEN', TIMESTAMP '2026-08-15 07:30:00', NULL, NULL),
    ('A011', 'F1012', 'AIR_TRAFFIC', 'P1', 'OPEN', TIMESTAMP '2026-08-15 08:35:00', NULL, NULL);

CREATE TABLE flight_sensors (
    sensor_id VARCHAR,
    flight_id VARCHAR,
    series_id VARCHAR,
    series_name VARCHAR,
    unit VARCHAR
);
INSERT INTO flight_sensors VALUES
    ('S-F1006-ALT', 'F1006', 'SER-F1006-ALT', 'altitude', 'ft'),
    ('S-F1006-SPD', 'F1006', 'SER-F1006-SPD', 'speed', 'knots'),
    ('S-F1008-ALT', 'F1008', 'SER-F1008-ALT', 'altitude', 'ft'),
    ('S-F1008-SPD', 'F1008', 'SER-F1008-SPD', 'speed', 'knots'),
    ('S-F1011-ALT', 'F1011', 'SER-F1011-ALT', 'altitude', 'ft'),
    ('S-F1011-SPD', 'F1011', 'SER-F1011-SPD', 'speed', 'knots'),
    ('S-F1012-ALT', 'F1012', 'SER-F1012-ALT', 'altitude', 'ft'),
    ('S-F1012-SPD', 'F1012', 'SER-F1012-SPD', 'speed', 'knots');

CREATE TABLE sensor_readings (
    series_id VARCHAR,
    ts TIMESTAMP,
    value_double DOUBLE
);
INSERT INTO sensor_readings VALUES
    ('SER-F1006-ALT', TIMESTAMP '2026-08-12 08:40:00', 12000),
    ('SER-F1006-ALT', TIMESTAMP '2026-08-12 09:00:00', 28500),
    ('SER-F1006-ALT', TIMESTAMP '2026-08-12 09:20:00', 31000),
    ('SER-F1006-SPD', TIMESTAMP '2026-08-12 08:40:00', 250),
    ('SER-F1006-SPD', TIMESTAMP '2026-08-12 09:00:00', 430),
    ('SER-F1006-SPD', TIMESTAMP '2026-08-12 09:20:00', 455),
    ('SER-F1008-ALT', TIMESTAMP '2026-08-14 08:00:00', 18000),
    ('SER-F1008-ALT', TIMESTAMP '2026-08-14 10:00:00', 35000),
    ('SER-F1008-ALT', TIMESTAMP '2026-08-14 12:00:00', 37000),
    ('SER-F1008-SPD', TIMESTAMP '2026-08-14 08:00:00', 310),
    ('SER-F1008-SPD', TIMESTAMP '2026-08-14 10:00:00', 470),
    ('SER-F1008-SPD', TIMESTAMP '2026-08-14 12:00:00', 482),
    ('SER-F1011-ALT', TIMESTAMP '2026-08-15 09:00:00', 14000),
    ('SER-F1011-ALT', TIMESTAMP '2026-08-15 09:20:00', 29500),
    ('SER-F1011-ALT', TIMESTAMP '2026-08-15 09:40:00', 32000),
    ('SER-F1011-SPD', TIMESTAMP '2026-08-15 09:00:00', 280),
    ('SER-F1011-SPD', TIMESTAMP '2026-08-15 09:20:00', 420),
    ('SER-F1011-SPD', TIMESTAMP '2026-08-15 09:40:00', 450),
    ('SER-F1012-ALT', TIMESTAMP '2026-08-15 10:00:00', 12000),
    ('SER-F1012-ALT', TIMESTAMP '2026-08-15 12:00:00', 30000),
    ('SER-F1012-ALT', TIMESTAMP '2026-08-15 14:00:00', 39000),
    ('SER-F1012-SPD', TIMESTAMP '2026-08-15 10:00:00', 260),
    ('SER-F1012-SPD', TIMESTAMP '2026-08-15 12:00:00', 445),
    ('SER-F1012-SPD', TIMESTAMP '2026-08-15 14:00:00', 490);
