# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.


from datus.utils.class_utils import get_public_instance_methods


class SampleBase:
    def public_method(self):
        pass

    def another_public(self):
        pass

    def _private_method(self):
        pass

    def __dunder_method(self):
        pass

    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass


class ChildClass(SampleBase):
    def child_method(self):
        pass


class EmptyClass:
    pass


class OnlyPrivate:
    def _hidden(self):
        pass

    def __very_hidden(self):
        pass


class TestGetPublicInstanceMethods:
    def test_returns_public_methods(self):
        methods = get_public_instance_methods(SampleBase)
        assert "public_method" in methods
        assert "another_public" in methods

    def test_excludes_private_methods(self):
        methods = get_public_instance_methods(SampleBase)
        assert "_private_method" not in methods

    def test_excludes_dunder_methods(self):
        methods = get_public_instance_methods(SampleBase)
        assert "__dunder_method" not in methods

    def test_excludes_static_methods(self):
        methods = get_public_instance_methods(SampleBase)
        assert "static_method" not in methods

    def test_excludes_class_methods(self):
        methods = get_public_instance_methods(SampleBase)
        assert "class_method" not in methods

    def test_returns_dict(self):
        result = get_public_instance_methods(SampleBase)
        assert isinstance(result, dict)

    def test_empty_class_returns_empty_dict(self):
        result = get_public_instance_methods(EmptyClass)
        assert result == {}

    def test_only_private_class_returns_empty_dict(self):
        result = get_public_instance_methods(OnlyPrivate)
        assert result == {}

    def test_child_class_includes_inherited_methods(self):
        methods = get_public_instance_methods(ChildClass)
        assert "child_method" in methods
        assert "public_method" in methods
        assert "another_public" in methods

    def test_child_class_excludes_inherited_private(self):
        methods = get_public_instance_methods(ChildClass)
        assert "_private_method" not in methods

    def test_values_are_callable(self):
        methods = get_public_instance_methods(SampleBase)
        for name, method in methods.items():
            assert callable(method), f"{name} should be callable"
