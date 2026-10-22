from tasa_droid.p2.result_centered_slicing import (
    InstructionRecord,
    MethodIR,
    SlicePolicy,
    slice_program,
)


ROLE_BY_REFERENCE = {
    "ldevice;->getid()ljava/lang/string;": {"device_identifier_access"},
    "lcrypto;->encrypt(ljava/lang/string;)ljava/lang/string;": {"cryptography"},
    "lnet;->send(ljava/lang/string;)v": {"network_transmission_or_upload"},
}


def map_roles(reference):
    return ROLE_BY_REFERENCE.get(reference, set())


def method(signature, *rows):
    return MethodIR(
        signature=signature,
        parameter_registers=tuple(),
        instructions=tuple(
            InstructionRecord(index * 2, name, output)
            for index, (name, output) in enumerate(rows)
        ),
    )


def test_real_register_flow_forms_multi_role_unit():
    program = [method(
        "lapp;->run()v",
        ("invoke-static", "LDevice;->getId()Ljava/lang/String;"),
        ("move-result-object", "v0"),
        ("invoke-static", "v0, LCrypto;->encrypt(Ljava/lang/String;)Ljava/lang/String;"),
        ("move-result-object", "v1"),
        ("invoke-static", "v1, LNet;->send(Ljava/lang/String;)V"),
        ("return-void", ""),
    )]
    result = slice_program(program, map_roles)
    units = result["units"]
    assert len(units) == 1
    assert units[0]["dependency_status"] == "RESOLVED_DEPENDENCY"
    assert units[0]["roles"] == [
        "cryptography",
        "device_identifier_access",
        "network_transmission_or_upload",
    ]
    assert "ARGUMENT_DEF_USE" in units[0]["evidence"]


def test_same_method_without_data_flow_remains_singleton():
    program = [method(
        "lapp;->run()v",
        ("invoke-static", "LDevice;->getId()Ljava/lang/String;"),
        ("move-result-object", "v0"),
        ("const-string", "v1, 'hello'"),
        ("invoke-static", "v1, LNet;->send(Ljava/lang/String;)V"),
        ("return-void", ""),
    )]
    result = slice_program(program, map_roles)
    units = result["units"]
    assert len(units) == 1
    assert units[0]["dependency_status"] == "UNRESOLVED_SINGLETON"
    assert units[0]["roles"] == ["network_transmission_or_upload"]


def test_shared_field_connects_cross_method_behavior():
    collect = method(
        "lapp;->collect()v",
        ("invoke-static", "LDevice;->getId()Ljava/lang/String;"),
        ("move-result-object", "v0"),
        ("sput-object", "v0, LApp;->globalId Ljava/lang/String;"),
        ("return-void", ""),
    )
    upload = method(
        "lapp;->upload()v",
        ("sget-object", "v0, LApp;->globalId Ljava/lang/String;"),
        ("invoke-static", "v0, LNet;->send(Ljava/lang/String;)V"),
        ("return-void", ""),
    )
    result = slice_program([collect, upload], map_roles, SlicePolicy(max_iterations=8))
    units = result["units"]
    resolved = [unit for unit in units if unit["dependency_status"] == "RESOLVED_DEPENDENCY"]
    assert len(resolved) == 1
    assert resolved[0]["roles"] == [
        "device_identifier_access",
        "network_transmission_or_upload",
    ]
    assert "FIELD_READ" in resolved[0]["evidence"]
    assert result["field_with_resolved_role_count"] == 1
