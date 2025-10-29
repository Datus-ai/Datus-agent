-- Please list the zip code of all the charter schools in Fresno County Office of Education.
-- Charter schools refers to `Charter School (Y/N)` = 1 in the table fprm
SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1;