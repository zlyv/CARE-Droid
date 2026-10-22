import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_result_centered_slice_dry_run_ag.py"
SPEC = importlib.util.spec_from_file_location("result_centered_driver", SCRIPT_PATH)
DRIVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DRIVER)


class FakeCode:
    def get_registers_size(self):
        return 1

    def get_ins_size(self):
        return 0


class FakeEncodedMethod:
    def get_class_name(self):
        return "Lapp/Main;"

    def get_name(self):
        return "run"

    def get_descriptor(self):
        return "()V"

    def get_code(self):
        return FakeCode()

    def get_instructions_idx(self):
        return []


class FakeMethodIdItem:
    def __init__(self, encoded):
        self.encoded = encoded

    def get_method(self):
        return self.encoded


class FakeDalvik:
    def __init__(self, method_id_item):
        self.method_id_item = method_id_item

    def get_methods(self):
        return [self.method_id_item]


def test_method_id_item_is_unwrapped_before_bytecode_access():
    encoded = FakeEncodedMethod()
    wrapped = FakeMethodIdItem(encoded)
    assert DRIVER.unwrap_encoded_method(wrapped) is encoded
    methods = DRIVER.build_method_ir(FakeDalvik(wrapped))
    assert len(methods) == 1
    assert methods[0].signature == "lapp/main;->run()v"
