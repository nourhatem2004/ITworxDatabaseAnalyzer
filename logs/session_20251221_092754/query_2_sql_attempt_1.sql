WITH RankedStudentScores AS (
    SELECT
        c.Name AS CourseName,
        su.FirstName AS StudentFirstName,
        su.LastName AS StudentLastName,
        ss.Mark,
        ROW_NUMBER() OVER (PARTITION BY ss.CourseId ORDER BY ss.Mark DESC) AS StudentRank
    FROM
        dbo.StudentScore AS ss
    INNER JOIN
        dbo.Courses AS c ON ss.CourseId = c.Id
    INNER JOIN
        dbo.Users AS u ON ss.StudentId = u.Id
    INNER JOIN
        dbo.SystemUsers AS su ON u.SystemUserId = su.Id
    WHERE
        ss.IsDeleted = 0
        AND c.IsDeleted = 0
)
SELECT
    CourseName,
    StudentFirstName,
    StudentLastName,
    Mark
FROM
    RankedStudentScores
WHERE
    StudentRank <= 3
ORDER BY
    CourseName ASC,
    StudentRank ASC;