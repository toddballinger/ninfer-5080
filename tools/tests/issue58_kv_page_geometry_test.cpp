#include "runtime/engine/issue58_kv_page_geometry.h"
#include <cassert>
#include <limits>
using namespace ninfer::runtime::issue58;
constexpr bool checks() {
    const KvPlaneGeometry page{KvPageOrder::PageMajor,4,4096,1024,0,0};
    auto p=kv_page_copy_span(page,2);
    if(!p || p->offset!=2048 || p->bytes_per_row!=1024 || p->rows!=1)return false;
    if(kv_page_copy_span(page,4))return false;
    const KvPlaneGeometry head{KvPageOrder::HeadMajor,4,4096,128,512,8};
    auto h=kv_page_copy_span(head,2);
    if(!h || h->offset!=256 || h->row_pitch!=512 || h->rows!=8 || h->bytes_per_row!=128)return false;
    auto bad=head;bad.row_pitch=256;
    if(kv_page_copy_span(bad,0))return false;
    bad=head;bad.rows=9;
    if(kv_page_copy_span(bad,0))return false;
    bad=head;bad.page_stride=0;
    if(kv_page_copy_span(bad,0))return false;
    bad=head;bad.page_stride=std::numeric_limits<std::size_t>::max();
    if(kv_page_copy_span(bad,2))return false;
    return true;
}
static_assert(checks());
int main(){assert(checks());}
