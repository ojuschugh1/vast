// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func external @test1 () -> !hl.int
int test1()
{
    // CHECK: hl.scope
    {
        int a = 0;
    }

    int a = 0;
    // CHECK: return [[C1:%[0-9]+]] : !hl.int
    return a;
}

// CHECK-LABEL: hl.func external @test2 ()
void test2()
{
    // CHECK: hl.scope
    // CHECK: hl.var "a" : !hl.lvalue<!hl.int>
    {
        int a;
    }

    // CHECK: hl.scope
    // CHECK: hl.var "a" : !hl.lvalue<!hl.int>
    {
        int a;
    }

    // CHECK: hl.scope
    // CHECK: hl.var "a" : !hl.lvalue<!hl.int>
    {
        int a;
    }
}

// CHECK-LABEL: hl.func external @test3 () -> !hl.int
int test3()
{
    // CHECK: hl.var "b" : !hl.lvalue<!hl.int>
    int b;

    // CHECK: hl.scope
    {
        // CHECK: hl.var "a" : !hl.lvalue<!hl.int>
        int a;
    }

    // CHECK-NOT: hl.scope
    int a;
    // CHECK: return [[C3:%[0-9]+]] : !hl.int
    return 0;
}

// CHECK-LABEL: hl.func external @test4 () -> !hl.int
int test4()
{
    // CHECK-NOT: hl.scope
    {
        int a = 0;
        // CHECK: hl.return
        return a;
    }
}
