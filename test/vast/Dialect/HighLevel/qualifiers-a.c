// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK: hl.var( i ): !hl.int
int i;

// CHECK: hl.var( u ): !hl<"unsigned int">
unsigned u;

// CHECK: hl.var( s ): !hl.int
signed s;

// CHECK: hl.var( ui ): !hl<"unsigned int">
unsigned int ui;

// CHECK: hl.var( us ): !hl<"unsigned short">
unsigned short us;

// CHECK: hl.var( ci ): !hl<"const int">
const int ci;

// CHECK: hl.var( cui ): !hl<"const unsigned int">
const unsigned cui;

// CHECK: hl.var( vi ): !hl<"volatile int">
volatile int vi;

// CHECK: hl.var( vui ): !hl<"volatile unsigned int">
volatile unsigned vui;

// CHECK: hl.var( cvi ): !hl<"const volatile int">
const volatile int cvi;

// CHECK: hl.var( cvi ): !hl<"const volatile unsigned int">
const volatile unsigned int cvi;

// CHECK: hl.var( b ): !hl.bool
bool b;

// CHECK: hl.var( cb ): !hl<"const bool">
const bool cb;

// CHECK: hl.var( cvb ): !hl<"const volatile bool">
const volatile bool cvb;
