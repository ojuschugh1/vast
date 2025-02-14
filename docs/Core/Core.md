<!-- Autogenerated by mlir-tblgen; don't manually edit -->
# 'core' Dialect

Utility dialect to provide common features for other dialects.
Dialect providing features that may be used by other dialects.
These features can be used by including "vast/Dialect/Core/Utils.td"
It also provides lazy.op for lazy evaluation of expressions and
binary logical operations that make use of it.

[TOC]

## Operation definition

### `core.bin.land` (::vast::core::BinLAndOp)

VAST core dialect logical binary operation


Syntax:

```
operation ::= `core.bin.land` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
```

Core dialect logical binary operation. This operation takes two operands
and returns one result, each of these is required to be of the same
type.

The custom assembly form of the operation is as follows:

%result = <op> %lhs, %rhs  : type

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | any type
| `rhs` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

### `core.bin.lor` (::vast::core::BinLOrOp)

VAST core dialect logical binary operation


Syntax:

```
operation ::= `core.bin.lor` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
```

Core dialect logical binary operation. This operation takes two operands
and returns one result, each of these is required to be of the same
type.

The custom assembly form of the operation is as follows:

%result = <op> %lhs, %rhs  : type

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | any type
| `rhs` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

### `core.lazy.op` (::vast::core::LazyOp)

Lazily evaluate a region.


Syntax:

```
operation ::= `core.lazy.op` $lazy attr-dict `:` type(results)
```

The operation serves to encapsulate delayed evaluation in its region.

Traits: NoTerminator

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

### `core.select` (::vast::core::SelectOp)

Select a value based on condition.


Syntax:

```
operation ::= `core.select` $cond `,` $thenRegion `,` $elseRegion attr-dict `:` functional-type(operands, results)
```

Usual select operation. First operand is selected if predicate is true, second
otherwise (to mirror how ternary works in C).

%result = <op> %cond %lhs, %rhs  : type

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `cond` | any type
| `thenRegion` | any type
| `elseRegion` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | any type

