from unittest.mock import patch

from datus.agent.evaluate import evaluate_result
from datus.agent.node import Node


class TestEvaluation:
    """Test suite for the evaluation module."""

    def test_evaluate_successful_result(self):
        """Test evaluation of a successful result."""
        # Create a mock node
        task = Node.new_instance(
            "task1",
            "Generate SQL",
            "generate_sql",
            "Find employees with salary over $50,000",
        )

        # Create a successful result
        result = "SELECT * FROM employees WHERE salary > 50000;"

        # Mock the evaluate_result function
        with patch(
            "agent.evaluation.evaluate.evaluate_result",
            return_value={"success": True, "suggestions": []},
        ):
            # Evaluate the result
            evaluation = evaluate_result(result, task)

            # Check the evaluation
            assert evaluation["success"] is True
            assert evaluation["suggestions"] == []

    def test_evaluate_failed_result(self):
        """Test evaluation of a failed result."""
        # Create a mock node
        task = Node.new_instance(
            "task1",
            "Generate SQL",
            "generate_sql",
            "Find employees with salary over $50,000",
        )

        # Create a failed result (syntax error in SQL)
        result = "SELECT * FORM employees WHERE salary > 50000;"

        # Mock the evaluate_result function
        with patch(
            "agent.evaluation.evaluate.evaluate_result",
            return_value={
                "success": False,
                "suggestions": ["Fix SQL syntax error: 'FORM' should be 'FROM'"],
            },
        ):
            # Evaluate the result
            evaluation = evaluate_result(result, task)

            # Check the evaluation
            assert evaluation["success"] is False
            assert len(evaluation["suggestions"]) == 1
            assert "Fix SQL syntax error" in evaluation["suggestions"][0]

    def test_evaluate_empty_result(self):
        """Test evaluation of an empty result."""
        # Create a mock node
        task = Node.new_instance(
            "task1",
            "Generate SQL",
            "generate_sql",
            "Find employees with salary over $50,000",
        )

        # Create an empty result
        result = ""

        # Mock the evaluate_result function
        with patch(
            "agent.evaluation.evaluate.evaluate_result",
            return_value={
                "success": False,
                "suggestions": ["No SQL query was generated"],
            },
        ):
            # Evaluate the result
            evaluation = evaluate_result(result, task)

            # Check the evaluation
            assert evaluation["success"] is False
            assert len(evaluation["suggestions"]) == 1
            assert "No SQL query was generated" in evaluation["suggestions"][0]

    def test_evaluate_error_result(self):
        """Test evaluation of a result containing an error message."""
        # Create a mock node
        task = Node.new_instance(
            "task1",
            "Execute SQL",
            "execute_sql",
            "SELECT * FROM employees WHERE salary > 50000;",
        )

        # Create a result with an error
        result = {"error": "Table 'employees' not found"}

        # Mock the evaluate_result function
        with patch(
            "agent.evaluation.evaluate.evaluate_result",
            return_value={
                "success": False,
                "suggestions": ["Create or verify the 'employees' table exists"],
            },
        ):
            # Evaluate the result
            evaluation = evaluate_result(result, task)

            # Check the evaluation
            assert evaluation["success"] is False
            assert len(evaluation["suggestions"]) == 1
            assert "Create or verify" in evaluation["suggestions"][0]

    def test_evaluate_different_task_types(self):
        """Test evaluation of results from different task types."""
        # Test SQL generation task
        sql_gen_task = Node.new_instance(
            "task1",
            "Generate SQL",
            "sql_generation",
            "Find employees with salary over $50,000",
        )
        sql_result = "SELECT * FROM employees WHERE salary > 50000;"

        # Test SQL execution task
        sql_exec_task = Node.new_instance(
            "task2",
            "Execute SQL",
            "sql_execution",
            "SELECT * FROM employees WHERE salary > 50000;",
        )
        exec_result = [{"id": 1, "name": "John Doe", "salary": 75000}]

        # Test query processing task
        query_task = Node.new_instance(
            "task3",
            "Process query",
            "query_processing",
            "Find employees with salary over $50,000",
        )
        query_result = {
            "entities": ["employees", "salary"],
            "conditions": [{"field": "salary", "operator": ">", "value": 50000}],
        }

        # Mock the evaluate_result function for each task type
        with patch("agent.evaluation.evaluate.evaluate_result") as mock_evaluate:
            # Configure mock to return different values based on task type
            def side_effect(result, task):
                if task.type == "sql_generation":
                    return {"success": True, "suggestions": []}
                elif task.type == "sql_execution":
                    return {"success": True, "suggestions": []}
                elif task.type == "query_processing":
                    return {"success": True, "suggestions": []}
                else:
                    return {"success": False, "suggestions": ["Unknown task type"]}

            mock_evaluate.side_effect = side_effect

            # Evaluate each result
            sql_gen_eval = evaluate_result(sql_result, sql_gen_task)
            sql_exec_eval = evaluate_result(exec_result, sql_exec_task)
            query_eval = evaluate_result(query_result, query_task)

            # Check that each evaluation was successful
            assert sql_gen_eval["success"] is True
            assert sql_exec_eval["success"] is True
            assert query_eval["success"] is True
