/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// AI-NIC RDMA status tolerance: a missing RDMA/ionic driver or empty result
// must not abort NIC discovery. No hardware required.

#include <gtest/gtest.h>

#include "amd_smi/impl/amd_smi_common.h"

namespace {

using amd::smi::nic_rdma_status_is_fatal;

TEST(NicUnit, RdmaOptionalStatusesAreNotFatal) {
  EXPECT_FALSE(nic_rdma_status_is_fatal(SMI_NIC_STATUS_SUCCESS));
  EXPECT_FALSE(nic_rdma_status_is_fatal(SMI_NIC_STATUS_NO_DATA));
  EXPECT_FALSE(nic_rdma_status_is_fatal(SMI_NIC_STATUS_DRIVER_NOT_LOADED));
}

TEST(NicUnit, RdmaOtherStatusesAreFatal) {
  for (smi_nic_status_t s :
       {SMI_NIC_STATUS_ERROR, SMI_NIC_STATUS_WRONG_PARAM, SMI_NIC_STATUS_NOT_FOUND,
        SMI_NIC_STATUS_NO_RESOURCE, SMI_NIC_STATUS_NOT_SUPPORTED, SMI_NIC_STATUS_NOT_INIT}) {
    EXPECT_TRUE(nic_rdma_status_is_fatal(s)) << "status=" << s;
  }
}

}  // namespace
