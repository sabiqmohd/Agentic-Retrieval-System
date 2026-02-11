"""Tools available to the LangGraph RAG workflow."""

import re
import operator
from typing import Dict, Any, Union


# Supported operators for safe evaluation
OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '//': operator.floordiv,
    '%': operator.mod,
    '^': operator.pow,
    '**': operator.pow,
}


def safe_calculator(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a simple arithmetic expression.
    
    Supports: +, -, *, /, //, %, ^ (power), ** (power)
    Handles: integers, floats, parentheses, negative numbers
    
    Args:
        expression: Arithmetic expression string, e.g., "5 + 3 * 2" or "(10 - 4) / 2"
        
    Returns:
        Dict with 'expression', 'result', and 'success' keys.
        On error, 'result' is None and 'error' contains the message.
        
    Examples:
        >>> safe_calculator("5 + 3")
        {'expression': '5 + 3', 'result': 8.0, 'success': True}
        
        >>> safe_calculator("10 / 0")
        {'expression': '10 / 0', 'result': None, 'success': False, 'error': 'Division by zero'}
    """
    original_expr = expression
    
    try:
        # Sanitize: only allow digits, operators, parentheses, spaces, decimal points
        sanitized = re.sub(r'[^\d\+\-\*\/\%\^\(\)\.\s]', '', expression)
        
        if not sanitized.strip():
            return {
                'expression': original_expr,
                'result': None,
                'success': False,
                'error': 'Empty or invalid expression'
            }
        
        # Replace ^ with ** for power operations
        sanitized = sanitized.replace('^', '**')
        
        # Validate structure: must have at least one number
        if not re.search(r'\d', sanitized):
            return {
                'expression': original_expr,
                'result': None,
                'success': False,
                'error': 'No numbers found in expression'
            }
        
        # Check for balanced parentheses
        paren_count = 0
        for char in sanitized:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if paren_count < 0:
                return {
                    'expression': original_expr,
                    'result': None,
                    'success': False,
                    'error': 'Unbalanced parentheses'
                }
        
        if paren_count != 0:
            return {
                'expression': original_expr,
                'result': None,
                'success': False,
                'error': 'Unbalanced parentheses'
            }
        
        # Use compile + eval with restricted globals for safety
        # Only allow numeric operations, no builtins
        code = compile(sanitized, '<string>', 'eval')
        
        # Check for disallowed names in bytecode
        for name in code.co_names:
            return {
                'expression': original_expr,
                'result': None,
                'success': False,
                'error': f'Invalid operation: {name}'
            }
        
        # Evaluate with empty namespace
        result = eval(code, {"__builtins__": {}}, {})
        
        # Handle infinity and NaN
        if isinstance(result, float):
            if result != result:  # NaN check
                return {
                    'expression': original_expr,
                    'result': None,
                    'success': False,
                    'error': 'Result is not a number (NaN)'
                }
            if abs(result) == float('inf'):
                return {
                    'expression': original_expr,
                    'result': None,
                    'success': False,
                    'error': 'Result is infinite (overflow)'
                }
        
        return {
            'expression': original_expr,
            'result': float(result),
            'success': True
        }
        
    except ZeroDivisionError:
        return {
            'expression': original_expr,
            'result': None,
            'success': False,
            'error': 'Division by zero'
        }
    except (SyntaxError, TypeError) as e:
        return {
            'expression': original_expr,
            'result': None,
            'success': False,
            'error': f'Invalid expression syntax: {str(e)}'
        }
    except Exception as e:
        return {
            'expression': original_expr,
            'result': None,
            'success': False,
            'error': f'Calculation error: {str(e)}'
        }
