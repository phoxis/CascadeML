from activation.activation_abs_c import activation_abs_c as activation_abs_c
from activation.activation_factory_c import activation_factory_c as activation_factory_c
import activation.functions
# FIXME: Inspect how packaging is done in reality
del activation.functions.activation_abs_c
